use ark_ec::{ AffineRepr };
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tonic::{transport::Server, Request, Response, Status};
use p2p;
use crate::p2p::pb::p2p_server::{P2p, P2pServer};
use p2p::pb::{ResponseMessage, RequestMessage};
use num_bigint::BigUint;
use ark_bls12_377::G1Affine;
use ark_bls12_377_latest::{ G1Affine as GAffine, G1Projective};
use ark_ff::fields::PrimeField;

// defining a struct for our service
#[derive(Default)]
pub struct MyP2p {}

// implementing grpc for service defined in .proto
#[tonic::async_trait]
impl P2p for MyP2p {
    // grpc implemented as function
    async fn send(&self,request:Request<RequestMessage>)->Result<Response<ResponseMessage>,Status>{
        let points = Vec::<BigUint>::deserialize_compressed(&*request.get_ref().points).unwrap_or(vec![]);
        let scalars = Vec::<BigUint>::deserialize_compressed(&*request.get_ref().scalars).unwrap_or(vec![]);
        let msm_result = ingo_x::msm_cloud::<G1Affine>(&points, &scalars).0;
        let proj_x_field = <GAffine as AffineRepr>::BaseField::from_le_bytes_mod_order(&msm_result[0].to_bytes_le());
        let proj_y_field = <GAffine as AffineRepr>::BaseField::from_le_bytes_mod_order(&msm_result[1].to_bytes_le());
        let proj_z_field = <GAffine as AffineRepr>::BaseField::from_le_bytes_mod_order(&msm_result[2].to_bytes_le());
        let point: GAffine = G1Projective::new(proj_x_field, proj_y_field, proj_z_field).into();
        let mut serialized_result = vec![];
        let _ = point.serialize_compressed(&mut serialized_result);
        let msg = if points.len() == 0 && scalars.len() == 0 { "Failure" } else { "Success" };
        // returning a response as ResponseMessage message as defined in .proto
        Ok(Response::new(ResponseMessage{
            // reading data from request which is awrapper around our RequestMessage message defined in .proto
            result: serialized_result,
            success: String::from(msg),
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // defining address for our service
    let addr = "[::1]:50051".parse().unwrap();
    // creating a service
    let p2p = MyP2p::default();
    println!("Server listening on {}", addr);
    // adding our service to our server.
    Server::builder()
        .add_service(P2pServer::new(p2p))
        .serve(addr)
        .await?;
    Ok(())
}