use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use duct::cmd;
use serde::Serialize;
use std::{
    fs::{self, File},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Reproducible cache/memory measurements
    Cache {
        /// Which tool to use
        #[arg(long, value_enum, default_value_t=Tool::Cachegrind)]
        tool: Tool,
        /// Dataset name (used by your builder)
        #[arg(long, default_value = "default")]
        dataset: String,
        /// Extra args passed to your binary
        #[arg(short)]
        target: usize,
        /// Extra args passed to your binary
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Tool {
    Cachegrind,
    Massif,
    MassifPages,
}

#[derive(Serialize)]
struct Meta {
    timestamp_utc: String,
    rustc: String,
    git_rev: String,
    tool: String,
    dataset: String,
    cmdline: Vec<String>,
    sysinfo: SysInfo,
}

#[derive(Serialize)]
struct SysInfo {
    cpu: String,
    governor: Option<String>,
    numa_nodes: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Cache {
            tool,
            dataset,
            target,
            args,
        } => run_cache(tool, dataset, target, args)?,
    }
    Ok(())
}

fn run_cache(
    tool: Tool,
    dataset: String,
    target: usize,
    args: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 0) Ensure artifacts dir
    let out = PathBuf::from("./.artifacts").join("measurements");
    fs::create_dir_all(&out)?;

    // 2) Resolve paths
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let outfilepath: &Path = dataset.as_ref();
    let outfilename = outfilepath
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> { "error getting out file name".into() })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error getting out file name str".into()
        })?;
    let prefix = out.join(format!("{}_{}_{}.{}", target, outfilename, stamp, "cg.out"));
    drop(File::create(prefix.as_path())?);

    // 3) Collect meta for reproducibility
    let rustc_v = String::from_utf8(cmd!("rustc", "--version").read()?.into())?;
    let git_rev = String::from_utf8(cmd!("git", "rev-parse", "HEAD").read()?.into())?;
    let cpu = String::from_utf8(cmd!("sh", "-c", "lscpu | head -n1").read()?.into())?;

    let meta = Meta {
        timestamp_utc: stamp.clone(),
        rustc: rustc_v.trim().to_string(),
        git_rev: git_rev.trim().to_string(),
        tool: format!("{tool:?}"),
        dataset: dataset.clone(),
        cmdline: args.clone(),
        sysinfo: SysInfo {
            cpu: cpu.trim().to_string(),
            governor: read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            numa_nodes: read_file("/sys/devices/system/node/possible"),
        },
    };
    fs::write(
        prefix.with_extension("meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;

    // 4) Run under the chosen tool
    match tool {
        Tool::Cachegrind => {
            let out = format!(
                "--cachegrind-out-file={}",
                prefix
                    .as_path()
                    .as_os_str()
                    .to_str()
                    .ok_or_else(|| -> Box<dyn std::error::Error> { "error".into() })?
            );
            cmd!(
                "valgrind",
                "--tool=cachegrind",
                &out,
                "./target/bench_cache/sparx",
                "-v",
                "-c",
                &target.to_string(),
                "-t8",
                "-m",
                "-f",
                &dataset
            )
            .run()?;
            // Optional: postprocess
            let txt = prefix.with_extension("cg.txt");
            let cg = cmd!("cg_annotate", &prefix).read()?;
            drop(File::create(txt.as_path())?);
            fs::write(txt, cg)?;
            println!("Results in {:?}", prefix);
        }
        Tool::Massif => {
            let outfile = prefix.with_extension("massif.out");
            let outfile = outfile
                .with_extension("massif.out")
                .canonicalize()
                .unwrap_or_else(|_| {
                    // fallback if the file doesn’t exist yet
                    std::env::current_dir().unwrap().join(&outfile)
                });
            if outfile.exists() {
                // Best-effort; if this fails, you’ll see a clear error here rather than from Valgrind
                fs::remove_file(&outfile)?;
            }
            if let Some(p) = outfile.parent() {
                fs::create_dir_all(p)?;
            }
            let f = std::fs::File::create(&outfile)?; // writable by you
            let mut perm = f.metadata()?.permissions();
            perm.set_mode(0o777);
            f.set_permissions(perm)?;
            drop(f); // close before launching valgrind
            let out = format!(
                "--massif-out-file={}",
                outfile
                    .as_path()
                    .as_os_str()
                    .to_str()
                    .ok_or_else(|| -> Box<dyn std::error::Error> { "error".into() })?
            );
            cmd!(
                "valgrind",
                "--tool=massif",
                "--time-unit=ms",
                &out,
                "./target/bench_cache/sparx",
                "-v",
                "-c",
                &target.to_string(),
                "-t8",
                "-m",
                "-f",
                &dataset
            )
            .run()?;
            // Optional: keep both ms_print and raw
            let txt = prefix.with_extension("massif.txt");
            let ms = cmd!("ms_print", &outfile).read()?;
            drop(File::create(txt.as_path())?);
            fs::write(txt, ms)?;
            println!("Results in {:?}", outfile);
        }
        Tool::MassifPages => {
            let outfile = prefix.with_extension("massif.out");
            let outfile = outfile
                .with_extension("massif.out")
                .canonicalize()
                .unwrap_or_else(|_| {
                    // fallback if the file doesn’t exist yet
                    std::env::current_dir().unwrap().join(&outfile)
                });
            if outfile.exists() {
                // Best-effort; if this fails, you’ll see a clear error here rather than from Valgrind
                fs::remove_file(&outfile)?;
            }
            if let Some(p) = outfile.parent() {
                fs::create_dir_all(p)?;
            }
            let f = std::fs::File::create(&outfile)?; // writable by you
            let mut perm = f.metadata()?.permissions();
            perm.set_mode(0o777);
            f.set_permissions(perm)?;
            drop(f); // close before launching valgrind
            let out = format!(
                "--massif-out-file={}",
                outfile
                    .as_path()
                    .as_os_str()
                    .to_str()
                    .ok_or_else(|| -> Box<dyn std::error::Error> { "error".into() })?
            );
            cmd!(
                "valgrind",
                "--tool=massif",
                "--time-unit=ms",
                "--pages-as-heap=yes",
                &out,
                "./target/bench_cache/sparx",
                "-v",
                "-c",
                &target.to_string(),
                "-t32",
                "-m",
                "-f",
                &dataset
            )
            .run()?;
            // Optional: keep both ms_print and raw
            let txt = prefix.with_extension("massif.txt");
            let ms = cmd!("ms_print", &outfile).read()?;
            drop(File::create(txt.as_path())?);
            fs::write(txt, ms)?;
            println!("Results in {:?}", outfile);
        }
    }
    Ok(())
}

fn read_file(p: &str) -> Option<String> {
    fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
