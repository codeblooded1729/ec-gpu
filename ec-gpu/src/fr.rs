use ark_ec::pairing::Pairing;

use crate::{GpuName, GpuField};

pub type Curve = ark_bls12_377::Bls12_377;
impl GpuName for <Curve as Pairing>::ScalarField {
    fn name() -> String {
        name!()
    }
}

impl GpuField for <Curve as Pairing>::ScalarField {
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
