use clap::Parser;
use tracing::level_filters::LevelFilter;

#[derive(Debug, Parser)]
pub struct Cli {
    /// OFF < ERROR < WARN < INFO < DEBUG < TRACE
    #[arg(short, long, default_value_t = LevelFilter::OFF)]
    pub log_level: LevelFilter,
    pub epochs: u64,
}
