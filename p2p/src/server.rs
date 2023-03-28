use tonic::{transport::Server, Request, Response, Status};
use p2p;
use crate::p2p::pb::p2p_server::{P2p, P2pServer};
use p2p::pb::{ResponseMessage, RequestMessage};

// defining a struct for our service
#[derive(Default)]
pub struct MyP2p {}

// implementing grpc for service defined in .proto
#[tonic::async_trait]
impl P2p for MyP2p {
    // grpc implemented as function
    async fn send(&self,request:Request<RequestMessage>)->Result<Response<ResponseMessage>,Status>{
        // returning a response as ResponseMessage message as defined in .proto
        Ok(Response::new(ResponseMessage{
            // reading data from request which is awrapper around our RequestMessage message defined in .proto
            message:format!("p2p - msm-calculate result {}",request.get_ref().name),
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