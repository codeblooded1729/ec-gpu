
use ark_ff::{One, PrimeField, Zero};
use ark_poly::EvaluationDomain;
use ark_std::{cfg_iter, cfg_iter_mut, vec};

use ec_gpu::GpuField;
use ec_gpu_gen::{
    fft::FftKernel,
    rust_gpu_tools::Device,
    threadpool::Worker,
};

use crate::Vec;
use ark_relations::r1cs::{
    ConstraintMatrices, ConstraintSystemRef, Result as R1CSResult, SynthesisError,
};
use core::ops::{AddAssign, Deref};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[inline]
/// Computes the inner product of `terms` with `assignment`.
pub fn evaluate_constraint<'a, LHS, RHS, R>(terms: &'a [(LHS, usize)], assignment: &'a [RHS]) -> R
where
    LHS: One + Send + Sync + PartialEq,
    RHS: Send + Sync + core::ops::Mul<&'a LHS, Output = RHS> + Copy,
    R: Zero + Send + Sync + AddAssign<RHS> + core::iter::Sum,
{
    // Need to wrap in a closure when using Rayon
    #[cfg(feature = "parallel")]
    let zero = || R::zero();
    #[cfg(not(feature = "parallel"))]
    let zero = R::zero();

    let res = cfg_iter!(terms).fold(zero, |mut sum, (coeff, index)| {
        let val = &assignment[*index];

        if coeff.is_one() {
            sum += *val;
        } else {
            sum += val.mul(coeff);
        }

        sum
    });

    // Need to explicitly call `.sum()` when using Rayon
    #[cfg(feature = "parallel")]
    return res.sum();
    #[cfg(not(feature = "parallel"))]
    return res;
}

/// Computes instance and witness reductions from R1CS to
/// Quadratic Arithmetic Programs (QAPs).
pub trait R1CSToQAP {
    /// Computes a QAP instance corresponding to the R1CS instance defined by `cs`.
    fn instance_map_with_evaluation<F: PrimeField, D: EvaluationDomain<F>>(
        cs: ConstraintSystemRef<F>,
        t: &F,
    ) -> Result<(Vec<F>, Vec<F>, Vec<F>, F, usize, usize), SynthesisError>;

    #[inline]
    /// Computes a QAP witness corresponding to the R1CS witness defined by `cs`.
    fn witness_map<F: PrimeField + GpuField, D: EvaluationDomain<F>>(
        prover: ConstraintSystemRef<F>,
    ) -> Result<Vec<F>, SynthesisError> {
        let matrices = prover.to_matrices().unwrap();
        let num_inputs = prover.num_instance_variables();
        let num_constraints = prover.num_constraints();

        let cs = prover.borrow().unwrap();
        let prover = cs.deref();

        let full_assignment = [
            prover.instance_assignment.as_slice(),
            prover.witness_assignment.as_slice(),
        ]
        .concat();

        Self::witness_map_from_matrices::<F, D>(
            &matrices,
            num_inputs,
            num_constraints,
            &full_assignment,
        )
    }

    /// Computes a QAP witness corresponding to the R1CS witness defined by `cs`.
    fn witness_map_from_matrices<F: PrimeField + GpuField, D: EvaluationDomain<F>>(
        matrices: &ConstraintMatrices<F>,
        num_inputs: usize,
        num_constraints: usize,
        full_assignment: &[F],
    ) -> R1CSResult<Vec<F>>;

    /// Computes the exponents that the generator uses to calculate base
    /// elements which the prover later uses to compute `h(x)t(x)/delta`.
    fn h_query_scalars<F: PrimeField, D: EvaluationDomain<F>>(
        max_power: usize,
        t: F,
        zt: F,
        delta_inverse: F,
    ) -> Result<Vec<F>, SynthesisError>;
}

/// Computes the R1CS-to-QAP reduction defined in [`libsnark`](https://github.com/scipr-lab/libsnark/blob/2af440246fa2c3d0b1b0a425fb6abd8cc8b9c54d/libsnark/reductions/r1cs_to_qap/r1cs_to_qap.tcc).
pub struct LibsnarkReduction;

impl R1CSToQAP for LibsnarkReduction {
    #[inline]
    #[allow(clippy::type_complexity)]
    fn instance_map_with_evaluation<F: PrimeField, D: EvaluationDomain<F>>(
        cs: ConstraintSystemRef<F>,
        t: &F,
    ) -> R1CSResult<(Vec<F>, Vec<F>, Vec<F>, F, usize, usize)> {
        let matrices = cs.to_matrices().unwrap();
        let domain_size = cs.num_constraints() + cs.num_instance_variables();
        let domain = D::new(domain_size).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();

        let zt = domain.evaluate_vanishing_polynomial(*t);

        // Evaluate all Lagrange polynomials
        let coefficients_time = start_timer!(|| "Evaluate Lagrange coefficients");
        let u = domain.evaluate_all_lagrange_coefficients(*t);
        end_timer!(coefficients_time);

        let qap_num_variables = (cs.num_instance_variables() - 1) + cs.num_witness_variables();

        let mut a = vec![F::zero(); qap_num_variables + 1];
        let mut b = vec![F::zero(); qap_num_variables + 1];
        let mut c = vec![F::zero(); qap_num_variables + 1];

        {
            let start = 0;
            let end = cs.num_instance_variables();
            let num_constraints = cs.num_constraints();
            a[start..end].copy_from_slice(&u[(start + num_constraints)..(end + num_constraints)]);
        }

        for (i, u_i) in u.iter().enumerate().take(cs.num_constraints()) {
            for &(ref coeff, index) in &matrices.a[i] {
                a[index] += &(*u_i * coeff);
            }
            for &(ref coeff, index) in &matrices.b[i] {
                b[index] += &(*u_i * coeff);
            }
            for &(ref coeff, index) in &matrices.c[i] {
                c[index] += &(*u_i * coeff);
            }
        }

        Ok((a, b, c, zt, qap_num_variables, domain_size))
    }

    fn witness_map_from_matrices<F: PrimeField + GpuField, D: EvaluationDomain<F>>(
        matrices: &ConstraintMatrices<F>,
        num_inputs: usize,
        num_constraints: usize,
        full_assignment: &[F],
    ) -> R1CSResult<Vec<F>> {
        // initialize the kernel 
        let worker = Worker::new();
        let _log_threads = worker.log_num_threads();
        let devices = Device::all();
        let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
        let mut kern = FftKernel::<F>::create(programs).expect("Cannot initialize kernel!");

        let domain =
            D::new(num_constraints + num_inputs).ok_or(SynthesisError::PolynomialDegreeTooLarge)?;
        let domain_size = domain.size();
        let zero = F::zero();

        let omega = omega::<F>(num_constraints + num_inputs);
        let omega_inv = omega.inverse().unwrap();
        let log2_n = ((num_constraints + num_inputs) as f32).log2().ceil() as u32;

        let mut a = vec![zero; domain_size];
        let mut b = vec![zero; domain_size];

        cfg_iter_mut!(a[..num_constraints])
            .zip(cfg_iter_mut!(b[..num_constraints]))
            .zip(cfg_iter!(&matrices.a))
            .zip(cfg_iter!(&matrices.b))
            .for_each(|(((a, b), at_i), bt_i)| {
                *a = evaluate_constraint(&at_i, &full_assignment);
                *b = evaluate_constraint(&bt_i, &full_assignment);
            });

        {
            let start = num_constraints;
            let end = start + num_inputs;
            a[start..end].clone_from_slice(&full_assignment[..num_inputs]);
        }

        // retrieve the coefficients of poly a, b
        // domain.ifft_in_place(&mut a);
        // domain.ifft_in_place(&mut b);
        ifft(&mut kern, &mut a, &omega_inv, log2_n);
        ifft(&mut kern, &mut b, &omega_inv, log2_n);

        let coset_domain = domain.get_coset(F::GENERATOR).unwrap();

        // find a(gamma * omega^i) and b(gamma * omega^i)
        // coset_domain.fft_in_place(&mut a);
        // coset_domain.fft_in_place(&mut b);
        gpu_coset_fft(&mut kern, &mut a, &omega, log2_n);
        gpu_coset_fft(&mut kern, &mut b, &omega, log2_n);

        // find ab(gamma * omega^i)
        // let mut ab = domain.mul_polynomials_in_evaluation_domain(&a, &b);
        // drop(a);
        // drop(b);
        let mut ab = mul_componentwise(a, b);

        let mut c = vec![zero; domain_size];
        cfg_iter_mut!(c[..num_constraints])
            .enumerate()
            .for_each(|(i, c)| {
                *c = evaluate_constraint(&matrices.c[i], &full_assignment);
            });

        // retrieve the coefficients of c
        // domain.ifft_in_place(&mut c);
        ifft(&mut kern, &mut c, &omega_inv, log2_n);

        // find c(gamma * omega^i)
        // coset_domain.fft_in_place(&mut c);
        gpu_coset_fft(&mut kern, &mut c, &omega, log2_n);

        // find Z(gamma) (same as Z(gamma * omega^i) for any i)
        let vanishing_polynomial_over_coset = domain
            .evaluate_vanishing_polynomial(F::GENERATOR)
            .inverse()
            .unwrap();
        
        // this is FFT of (ab(gamma*X) - C(gamma*X))/Z(gamma * X)
        cfg_iter_mut!(ab).zip(c).for_each(|(ab_i, c_i)| {
            *ab_i -= &c_i;
            *ab_i *= &vanishing_polynomial_over_coset;
        });

        // retrive coeffcients of quotient poly
        // coset_domain.ifft_in_place(&mut ab);
        gpu_coset_ifft(&mut kern, &mut ab, &omega_inv, log2_n);

        Ok(ab)
    }

    fn h_query_scalars<F: PrimeField, D: EvaluationDomain<F>>(
        max_power: usize,
        t: F,
        zt: F,
        delta_inverse: F,
    ) -> Result<Vec<F>, SynthesisError> {
        let scalars = cfg_into_iter!(0..max_power)
            .map(|i| zt * &delta_inverse * &t.pow([i as u64]))
            .collect::<Vec<_>>();
        Ok(scalars)
    }
}

// /// performs fft over f(gamma * X) where gamma is taken to be generator of the field
// pub fn gpu_coset_fft<Fr: GpuField + Field + ff_PrimeField>(kern: &FftKernel<'_, Fr>, coeffs: &mut Vec<Fr>, omega: &Fr, log_n: u32){
//     let gen  = Fr::multiplicative_generator();
//     coeffs.iter_mut().enumerate().map(|(i, c)| { *c *= gen.pow_vartime([i as u64]); c} ).collect::<Vec<&mut Fr>>();
//     kern.radix_fft(coeffs, omega, log_n);
// }

// /// Convert Vec<F> into Vec<Fr>
// pub fn convert_to_fr<Fr: ff_PrimeField, F: PrimeField + BigInteger>(coeff: Vec<F>) -> Vec<Fr> {
//     coeff.iter().map(|e| Fr::from_repr(e.into_bigint().try_into().unwrap())).collect()
// }

// /// Convert Vec<Fr> into Vec<F>
// pub fn convert_to_f<Fr: ff_PrimeField, F: PrimeField + BigInteger>(coeff: Vec<Fr>) -> Vec<F> {
//     coeff.iter.map(|e| F::from(e.into_repr()));
// }
// /// Convert from coset to actual coeff
// pub fn invert_coset<Fr: ff_PrimeField>(coeffs: &mut Vec<Fr>){
//     let gen = Fr::multiplicative_generator();
//     coeffs.iter_mut().enumerate().map(|(i, c)| {*c *= gen.pow_vartime([i as u64]).invert().unwrap(); c}).collect::<&mut Vec<Fr>>();
// }

/// Perform inverse FFT over GPU
pub fn ifft<F: GpuField + PrimeField>(kern: &mut FftKernel<'_, F>, coeffs: &mut Vec<F>, omega: &F, log_n: u32){
    let _ = kern.radix_fft(coeffs, omega, log_n);
    let inv_n = F::from((2 as u64).pow(log_n)).inverse().unwrap();
    coeffs.iter_mut().for_each(|c| *c *= inv_n);
}

/// Perform coset FFT over GPU
pub fn gpu_coset_fft<F: GpuField + PrimeField>(kern: &mut FftKernel<'_, F>, coeffs: &mut Vec<F>, omega: &F, log_n: u32){
    let gen  = F::GENERATOR;
    let _ = coeffs.iter_mut().enumerate().map(|(i, c)| { *c *= gen.pow([i as u64]); c} ).collect::<Vec<&mut F>>();
    let _ = kern.radix_fft(coeffs, omega, log_n);
}

/// Perform coset IFFT over GPU
pub fn gpu_coset_ifft<F: GpuField + PrimeField>(kern: &mut FftKernel<'_, F>, coeffs: &mut Vec<F>, omega: &F, log_n: u32){
    let gen_inv  = F::GENERATOR.inverse().unwrap();
    ifft(kern, coeffs, omega, log_n);
    let _ = coeffs.iter_mut().enumerate().map(|(i, c)| { *c *= gen_inv.pow([i as u64]); c} ).collect::<Vec<&mut F>>();

}

/// find 2^n th root of unity where n is min such that 2^n > num_coeffs
fn omega<F: PrimeField>(num_coeffs: usize) -> F {
    // Compute omega, the 2^exp primitive root of unity
    let exp = (num_coeffs as f32).log2().ceil() as u32;
    let pow = (2 as u64).pow(exp);
    let omega = F::get_root_of_unity(pow).unwrap();
    omega
}

/// Multiply componentwise
fn mul_componentwise<F: PrimeField>(mut a: Vec<F>, b: Vec<F>) -> Vec<F> {
    assert_eq!(a.len(), b.len());
    ark_std::cfg_iter_mut!(a)
        .zip(b)
        .for_each(|(a_i, b_i)| *a_i *= b_i);

    a
}

#[test]
fn test_gpu_coset_fft(){
    use ark_ff::UniformRand;
    use ark_ff::Field;
    use ark_ec::pairing::Pairing;
    use ec_gpu::fr::Curve;
    type Fr = <Curve as Pairing>::ScalarField;
    let mut rng = rand::thread_rng();

    let worker = Worker::new();
    let _log_threads = worker.log_num_threads();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");
    let d = 16;
    let log_n = 4;
    let mut coeffs = (0..d).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
    let mut before = coeffs.clone();
    let omega = omega::<Fr>(coeffs.len());
    let omega_inv = omega.inverse().unwrap();
    gpu_coset_fft(&mut kern, &mut coeffs, &omega, log_n);
    gpu_coset_ifft(&mut kern, &mut coeffs, &omega_inv, log_n);
    
    assert!(coeffs == before);
}

