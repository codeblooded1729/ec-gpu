use p2p;
use crate::p2p::pb::p2p_client::P2pClient;
use p2p::pb::RequestMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
// creating a channel ie connection to server
    let channel = tonic::transport::Channel::from_static("http://[::1]:50051")
        .connect()
        .await?;
// creating gRPC client from channel
    let mut client = P2pClient::new(channel);
// creating a new Request
    let request = tonic::Request::new(
        RequestMessage {
            name:String::from("msm-calculate")
        },
    );
// sending request and waiting for response
    let response = client.send(request).await?.into_inner();
    println!("RESPONSE={:?}", response);
    Ok(())
}