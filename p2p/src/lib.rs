pub mod pb {
    // this one includes p2p.proto
    tonic::include_proto!("p2p_tonic");
}

pub mod funcs {
    use crate::pb::RequestMessage;
    use crate::pb::p2p_client::P2pClient;
    pub async fn get_res_from_cloud(points: Vec<u8>, scalars: Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let channel = tonic::transport::Channel::from_static("http://100.25.26.176:50051")
            .connect()
            .await?;
        // creating gRPC client from channel
        let mut client = P2pClient::new(channel);
        // creating a new Request
        let request = tonic::Request::new(
            RequestMessage {
                points,
                scalars,
            },
        );
    // sending request and waiting for response
        let response = client.send(request).await?.into_inner();
        Ok(response.result)
    }
}