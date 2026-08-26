use crate::routes;
use anyhow::{Result, ensure};
use axpindexer_core::Database;
use std::path::Path;
use tiny_http::{Header, Response, Server};
pub fn serve(db_path: &Path, host: &str, port: u16) -> Result<()> {
    ensure!(
        host == "127.0.0.1" || host == "localhost" || host == "::1",
        "refusing non-loopback host"
    );
    let server =
        Server::http(format!("{host}:{port}")).map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("Listening on http://{host}:{port}");
    for request in server.incoming_requests() {
        let db = Database::open(db_path)?;
        let (status, content, body) = routes::route(&db, request.url());
        let header = Header::from_bytes("Content-Type", content).expect("static header");
        request.respond(
            Response::from_string(body)
                .with_status_code(status)
                .with_header(header),
        )?
    }
    Ok(())
}
