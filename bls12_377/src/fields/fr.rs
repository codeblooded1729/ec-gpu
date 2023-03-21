//! Bls12-377 scalar field.
use std::{ops::{Neg, Add, Sub, Mul, AddAssign, SubAssign, MulAssign}, iter::{Sum, Product}, borrow::Borrow};
use ark_std::{ Zero, One, UniformRand};
use ark_ff::{Field, PrimeField, BigInteger, FftField};
/// Roots of unity computed from modulus and R using this sage code:
///
/// ```ignore
/// q = 8444461749428370424248824938781546531375899335154063827935233455917409239041
/// R = 6014086494747379908336260804527802945383293308637734276299549080986809532403 # Montgomery R
/// s = 47
/// o = q - 1
/// F = GF(q)
/// g = F.multiplicative_generator()
/// g = F.multiplicative_generator()
/// assert g.multiplicative_order() == o
/// g2 = g ** (o/2**s)
/// assert g2.multiplicative_order() == 2**s
/// def into_chunks(val, width, n):
///     return [int(int(val) // (2 ** (width * i)) % 2 ** width) for i in range(n)]
/// print("Gen: ", g * R % q)
/// print("Gen: ", into_chunks(g * R % q, 64, 4))
/// print("2-adic gen: ", into_chunks(g2 * R % q, 64, 4))
/// ```
use ark_ff::{fields::{Fp256, MontBackend, MontConfig}};
use subtle::{ConditionallySelectable, Choice, ConstantTimeEq, CtOption};
use crate::*;

#[derive(MontConfig)]
#[modulus = "8444461749428370424248824938781546531375899335154063827935233455917409239041"]
#[generator = "22"]
pub struct FrConfig;
pub type Fr = Fp256<MontBackend<FrConfig, 4>>;
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
pub struct Scalar(Fr);

impl Neg for Scalar {
    type Output = Scalar;

    #[inline]
    fn neg(self) -> Scalar {
        Self(self.0.neg())
    }
}

impl Add<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn add(self, rhs: &Scalar) -> Scalar {
        let mut out = *self;
        out += rhs;
        out
    }
}

impl Sub<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn sub(self, rhs: &Scalar) -> Scalar {
        let mut out = *self;
        out -= rhs;
        out
    }
}

impl Mul<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn mul(self, rhs: &Scalar) -> Scalar {
        let mut out = *self;
        out *= rhs;
        out
    }
}

impl AddAssign<&Scalar> for Scalar {
    #[inline]
    fn add_assign(&mut self, rhs: &Scalar) {
        self.0 += rhs.0;
    }
}

impl SubAssign<&Scalar> for Scalar {
    #[inline]
    fn sub_assign(&mut self, rhs: &Scalar) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<&Scalar> for Scalar {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        self.0 *= rhs.0;
    }
}

impl<T> Sum<T> for Scalar
where
    T: Borrow<Scalar>,
{
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.fold(<Scalar as ff::Field>::zero(), |sum, x| sum + x.borrow())
    }
}

impl<T> Product<T> for Scalar
where
    T: Borrow<Scalar>,
{
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = T>,
    {
        iter.fold(<Scalar as ff::Field>::one(), |product, x| product * x.borrow())
    }
}

impl ConditionallySelectable for Scalar {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        if choice.unwrap_u8() == 0 {
            *a
        }
        else{
            *b
        }
    }
}

impl ConstantTimeEq for Scalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        if self.0 == other.0 {
            Choice::from(1)
        }
        else{
            Choice::from(0)
        }
    }
}


crate::impl_add_sub!(Scalar);
crate::impl_add_sub_assign!(Scalar);
crate::impl_mul!(Scalar);
crate::impl_mul_assign!(Scalar);

impl ff::Field for Scalar {
    fn zero() -> Self { 
        Self(Fr::zero())
    }

    fn one() -> Self {
        Self(Fr::one())
    }

    fn random(mut rng: impl ark_std::rand::RngCore) -> Self {
        Self(Fr::rand(&mut rng))
    }

    fn square(&self) -> Self {
        Self(self.0.square())
    }

    fn double(&self) -> Self {
        Self(self.0.double())
    }

    fn invert(&self) -> CtOption<Self> {
        if self.0 == Fr::zero() {
            CtOption::new(Self::zero(), Choice::from(0))
        }
        else{
            CtOption::new(Self(self.0.inverse().unwrap()), Choice::from(1))
        }
    }

    fn sqrt(&self) -> subtle::CtOption<Self> {
        let elem = self.0;
        if let Some(sqrt) = elem.sqrt() {
            let choice = Choice::from(1);
            CtOption::new(Self(sqrt), choice)
        }
        else{
            let choice = Choice::from(0);
            CtOption::new(Self::zero(), choice)
        }
    }
}

impl From<u64> for Scalar{
    fn from(value: u64) -> Self {
        Self(value.into())
    }
}

impl ff::PrimeField for Scalar {
    // Little-endian non-Montgomery form bigint mod p.
    type Repr = [u8; 32];

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        CtOption::new(Self(Fr::from_le_bytes_mod_order(&repr)), Choice::from(1))
    }

    /// Converts a Montgomery form `Scalar` into little-endian non-Montgomery from.
    fn to_repr(&self) -> Self::Repr {
        self.0.0.to_bytes_le().try_into().unwrap()
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }

    const NUM_BITS: u32 = 255;

    const CAPACITY: u32 = Self::NUM_BITS - 1;

    fn multiplicative_generator() -> Self {
        Self(<Fr as FftField>::GENERATOR)
    }

    const S: u32 = 47;

    fn root_of_unity() -> Self {
        Self(<Fr as FftField>::TWO_ADIC_ROOT_OF_UNITY)
    }
}
impl ec_gpu::GpuName for Scalar {
    fn name() -> String {
        ec_gpu::name!()
    }
}

impl ec_gpu::GpuField for Scalar {
    fn one() -> Vec<u32> {
        vec![4294967283, 2099019775, 1879048178, 1918366991, 1361842158, 383260021, 733715101, 223074866]
    }

    fn r2() -> Vec<u32> {
        vec![3093398907, 634746810, 2288015647, 3425445813, 3856434579, 2815164559, 4025600313, 18864871]
    }

    fn modulus() -> Vec<u32> {
        vec![1, 168919040, 3489660929, 1504343806, 1547153409, 1622428958, 2586617174, 313222494]
    }
}

