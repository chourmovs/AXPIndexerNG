mod cli;
mod routes;
mod server;
use anyhow::Result;
use axpindexer_core::{Database, SearchQuery};
use clap::Parser;
use cli::{Cli, Command};
fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    match Cli::parse().command {
        Command::Health { db } => {
            let h = Database::open(db)?.health()?;
            println!(
                "AXPIndexerClient health OK\nSQLite: {}\nSchema: {}\nFTS5: OK",
                h.sqlite, h.schema
            )
        }
        Command::Search { db, query, limit } => {
            for r in Database::open(db)?.search(SearchQuery {
                text: &query,
                limit,
            })? {
                println!("{}\t{}\t{}", r.score, r.path, r.snippet)
            }
        }
        Command::Serve { db, host, port } => server::serve(&db, &host, port)?,
    }
    Ok(())
}
