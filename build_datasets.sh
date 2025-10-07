#!/usr/bin/env bash
set -euo pipefail

# Fixed configuration specs
URLS_FILE="dataset_urls.txt"
OUTPUT_PREFIX="ggcat_graph"

# Tunables that still make sense to keep as vars
MAINDIR=$(pwd)
WORKDIR="datasets"
THREADS="$(getconf _NPROCESSORS_ONLN || echo 8)"
COLORED=0
KEEP_GZ=0 # unzip by default

missing=0

need_cmd() {
	if ! command -v "$1" >/dev/null 2>&1; then
		echo "ERROR: '$1' is not installed or not on PATH." >&2
		missing=1
	fi
}

need_cmd python3
need_cmd cargo
need_cmd rustc

if ((missing)); then
	echo
	echo "Please install the missing tools and re-run this script."
	echo
	echo "Ubuntu example:"
	echo "  sudo apt update"
	echo "  sudo apt install -y python3 python3-pip python3-venv rustc cargo"
	echo
	echo "Verify afterwards with:"
	echo "  python3 --version"
	echo "  rustc --version"
	echo "  cargo --version"
	exit 1
fi

echo "[OK] Using: $(python3 --version 2>&1)"
echo "[OK] Using: $(cargo --version 2>&1)"
echo "[OK] Using: $(rustc --version 2>&1)"

usage() {
	cat <<'EOF'
Usage: build_graphs.sh [options]

Options:
  -d DIR     Working directory (default: data)
  -j INT     Threads (default: detected cores)
  -c         Build COLORED graphs (default: off)
  -Z         Keep compressed (.gz) inputs; do not gunzip (default: off)
  -h         Show help

Behavior (fixed):
  * Read dataset URLs from "dataset_urls.txt" (one per line; '#' for comments).
  * Build per-dataset graphs:
       - k=5 and k=10 for ALL datasets
       - k=15 for the FIRST TWO datasets listed
  * ggcat is built with default features (no feature flags)
  * Output prefix is fixed to "ggcat_graph"

Outputs:
  data/
    <downloaded files...>
    ggcat_graph.<basename>.k5.fasta.lz4
    ggcat_graph.<basename>.k10.fasta.lz4
    ggcat_graph.<basename>.k15.fasta.lz4 (first two datasets only)
EOF
}

while getopts ":d:j:Zch" opt; do
	case "$opt" in
	d) WORKDIR="$OPTARG" ;;
	j) THREADS="$OPTARG" ;;
	Z) KEEP_GZ=1 ;;
	c) COLORED=1 ;;
	h)
		usage
		exit 0
		;;
	\?)
		echo "Unknown option: -$OPTARG" >&2
		usage
		exit 2
		;;
	:)
		echo "Option -$OPTARG requires an argument." >&2
		usage
		exit 2
		;;
	esac
done

# 1) Ensure tools
need() { command -v "$1" >/dev/null 2>&1 || {
	echo "Missing required tool: $1" >&2
	exit 1
}; }
need curl
need git

# 2) Check URL list
if [[ ! -s "${URLS_FILE}" ]]; then
	echo "ERROR: ${URLS_FILE} not found or empty. Put one URL per line." >&2
	exit 1
fi

# 3.1) Create workdir
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
# 3.2) Create graphs directory
mkdir -p "graphs"
# 3.2) Create data directory
mkdir -p "data"
cd "data"

echo "Change working directory into: $(pwd)"

# 4) Get datasets

# Add empty input (always included in builds)
touch ___empty_input___.contigs.fa
FILES=("___empty_input___.contigs.fa")

# Now download datasets and append to FILES
echo "Downloading datasets listed in: ../../${URLS_FILE}"
while IFS= read -r url || [[ -n "$url" ]]; do
	[[ -z "$url" || "$url" =~ ^# ]] && continue
	fname="$(basename "${url}")"
	curl -L --fail --retry 5 --retry-connrefused --retry-delay 3 -o "${fname}" "${url}"
	if [[ "${KEEP_GZ}" -eq 0 && "${fname}" =~ \.gz$ ]]; then
		echo "Decompressing ${fname}"
		gunzip -f "${fname}"
		fname="${fname%.gz}"
	fi
	FILES+=("$fname")
done <"../../${URLS_FILE}"

if [[ ${#FILES[@]} -eq 0 ]]; then
	echo "No files downloaded. Check your URL list." >&2
	exit 1
fi

# back to workdir
cd ".."
# back to maindir
cd ".."
# into parentdir
cd ".."

# 5) Install Rust + ggcat if needed (default features only)
if ! command -v ggcat >/dev/null 2>&1; then
	echo "Installing Rust toolchain if needed..."
	if ! command -v cargo >/dev/null 2>&1; then
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
		# shellcheck disable=SC1090
		source "$HOME/.cargo/env"
		rustup toolchain install stable
	fi

	echo "Cloning and installing ggcat (default features)..."
	if [[ ! -d ggcat ]]; then
		git clone https://github.com/algbio/ggcat
	fi
	pushd ggcat >/dev/null
	cargo install --path crates/cmdline/ --locked
	popd >/dev/null
	echo "GGCAT installed to \$HOME/.cargo/bin. Ensure it's on your PATH."
fi

# 6) Helper: build a graph for one file and one k
build_one() {
	cd "./ggcat"

	local infile="$1"
	local k="$2"
	local idx="$3"

	local base
	base="$(basename "${infile}")"
	base="${base%.*}" # strip one extension
	local out="${MAINDIR}/${WORKDIR}/graphs/graph_${idx}_${k}.lz4"

	# # Prepare a temp list file for ggcat
	# local list="one_input_${base}.txt"
	# printf "%s\n" "$(realpath "${infile}")" >"${list}"
	# #
	# # Optional coloring: simple mapping (color name = sample base)
	# local cmd=(cargo run --release build -k "${k}" -j "${THREADS}" -c -e -l "${list}" --output-file "${out}")
	# if [[ "${COLORED}" -eq 1 ]]; then
	# 	local cmap="colormap_${base}.tsv"
	# 	printf "%s\t%s\n" "${base}" "$(realpath "${infile}")" >"${cmap}"
	# 	cmd+=(-c -d "${cmap}")
	# fi
	#
	local cmd=(cargo run --release build -k "${k}" -j "${THREADS}" -c -e "${infile}" --output-file "${out}")
	if [[ "${COLORED}" -eq 1 ]]; then
		local cmap="colormap_${base}.tsv"
		printf "%s\t%s\n" "${base}" "$(realpath "${infile}")" >"${cmap}"
		cmd+=(-c -d "${cmap}")
	fi

	printf "\n\n"
	echo "GGCAT output in ${out} from ${infile}"
	"${cmd[@]}"
	printf "\n\n"

	cd ".."
}

# 7) Build per dataset:
#    - k=5 and k=10 for all
#    - k=15 for the first two datasets in FILES
k15_limit=2
idx=0
echo "${FILES[@]}"
for f in "${FILES[@]}"; do
	# build_one "${f}" 5 "$idx"
	# build_one "${f}" 10 "$idx"
	# if ((idx < k15_limit)); then
	# 	build_one "${f}" 15 "$idx"
	# fi
	# ((idx++))
	printf -- '-> building idx=%d file=%s\n' "$idx" "$f"

	set +e
	build_one "$f" 5 "$idx"
	rc5=$?
	build_one "$f" 10 "$idx"
	rc10=$?
	rc15=-100000000
	if ((idx == 8 || idx == 9)); then
		build_one "$f" 15 "$idx"
		rc15=$?
	fi
	set -e

	if [[ "$rc15" == "NA" ]]; then rctext="k5=$rc5 k10=$rc10"; else rctext="k5=$rc5 k10=$rc10 k15=$rc15"; fi
	if ((rc5 != 0 || rc10 != 0 || (rc15 != -100000000 && rc15 != 0))); then
		echo "WARNING: build failures for idx=$idx file=$f :: $rctext"
	fi

	((++idx))
done

cd "${MAINDIR}"

echo "Building synthetic datasets using \`gen_dgd_mtx.py\`"

# ~16k protein edges (exact 15980)
python3 gen_dbg_mtx.py --alphabet protein20 --k 2 --out ./${WORKDIR}/graphs/synth_gen_dbg_protein20_k2_edges16k.mtx

# ~130k DNA edges (exact 131068)
python3 gen_dbg_mtx.py --alphabet dna --k 7 --out ./${WORKDIR}/graphs/synth_gen_dbg_dna_k7_edges130k.mtx

# ~8.4M DNA edges (exact 8,388,604)
python3 gen_dbg_mtx.py --alphabet dna --k 10 --out ./${WORKDIR}/graphs/synth_dna_k10_edges8_4M.mtx

# ~33.5M DNA edges (exact 33,554,428)
python3 gen_dbg_mtx.py --alphabet dna --k 11 --out ./${WORKDIR}/graphs/synth_dna_k11_edges33_5M.mtx

# ~128M protein edges (exact 127,999,980)
python3 gen_dbg_mtx.py --alphabet protein20 --k 5 --out ./${WORKDIR}/graphs/synth_prot20_k5_edges128M.mtx

# ~296M protein edges (exact 296,071,755)
python3 gen_dbg_mtx.py --alphabet protein23 --k 5 --out ./${WORKDIR}/graphs/synth_prot23_k5_edges296M.mtx

# ~6.8B protein edges (exact 6,809,650,871) **WARNING!!!: Gigabyte A1 took 1 hour to parse the resulting file into a graph.**
python3 gen_dbg_mtx.py --alphabet protein23 --k 6 --out ./${WORKDIR}/graphs/synth_prot23_k6_edges6_8B.mtx

echo
echo "All graphs built under: $(pwd)/${WORKDIR}/graphs"
