
use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use num_bigint::BigUint;
use p2p;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ark_std::UniformRand;
use ark_ec::AffineRepr;
use ark_ff::fields::Field;
use ark_ff::PrimeField;
use ec_gpu::fr::Curve;
use ark_bls12_377_latest::g1::G1Affine;
use ark_ec::CurveGroup;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // generate the data
    let (points, scalars, answer) = generate_data::<Curve>();

    let res = p2p::funcs::get_res_from_cloud(points, scalars).await?; 

    let unserialized_res = G1Affine::deserialize_compressed(&*res).unwrap();



    println!("cloud msm is {:?}", unserialized_res);
    println!("actual msm is {:?}", answer);

    Ok(())
}



pub fn generate_data<E: Pairing>() -> (Vec<u8>, Vec<u8>, E::G1Affine){
    let mut rng = ChaCha20Rng::from_entropy();
    let points = vec![E::G1Affine::rand(&mut rng), E::G1Affine::rand(&mut rng)];
    let scalars = vec![E::ScalarField::rand(&mut rng), E::ScalarField::rand(&mut rng)];

    let mut msm = E::G1Affine::zero().into_group();
    for i in 0..points.len(){
        msm += points[i] * scalars[i]; 
    }

    let mut points_as_big_int: Vec<BigUint> = vec![];
    let mut scalar_as_big_int: Vec<BigUint> = vec![];

    for i in 0..points.len(){
        if points[i].is_zero(){
            continue;
        }
        points_as_big_int.push(points[i]
                    .y()
                    .unwrap()
                    .to_base_prime_field_elements()
                    .collect::<Vec<_>>()[0]
                    .into_bigint()
                    .into()
                );
        points_as_big_int.push(points[i]
                    .x()
                    .unwrap()
                    .to_base_prime_field_elements()
                    .collect::<Vec<_>>()[0]
                    .into_bigint()
                    .into()
                );
        scalar_as_big_int.push(scalars[i].into_bigint().into());
    }
    let mut points_bytes = vec![];
    let mut scalar_bytes = vec![];
    points_as_big_int.serialize_compressed(&mut points_bytes).unwrap();
    scalar_as_big_int.serialize_compressed(&mut scalar_bytes).unwrap();

    (points_bytes, scalar_bytes, msm.into_affine())
}