#!/usr/bin/env bash
# ==============================================================================
# Renvi / ViTrans - Native MoE LLM Server Launcher (Standardized & Low-Overhead)
# Architecture: Qwen3.6-35B-A3B-UD-Q3_K_M (MoE, ~3B active params / token)
# Platform    : AMD Ryzen 7 8845H (8C/16T Zen 4) + RTX 4060 Laptop 8GB + 16GB RAM
# Optimization: 1) Maximum GPU offload (-fit on, FlashAttention on, KV q8_0)
#               2) Zero-SSD wear (-lzm off, -lm mmap, no disk checkpoints)
#               3) Low CPU heat (Physical core pinning, --poll 0 sleep mode)
# ==============================================================================
set -euo pipefail

# ANSI color codes
CLR_RESET='\033[0m'
CLR_BOLD='\033[1m'
CLR_RED='\033[0;31m'
CLR_GREEN='\033[0;32m'
CLR_YELLOW='\033[1;33m'
CLR_BLUE='\033[0;34m'
CLR_CYAN='\033[0;36m'
CLR_GRAY='\033[0;90m'

msg_info()  { printf "${CLR_CYAN}${CLR_BOLD}[INFO]${CLR_RESET} %s\n" "$*"; }
msg_ok()    { printf "${CLR_GREEN}${CLR_BOLD}[OK]${CLR_RESET}   %s\n" "$*"; }
msg_warn()  { printf "${CLR_YELLOW}${CLR_BOLD}[WARN]${CLR_RESET} %s\n" "$*"; }
msg_err()   { printf "${CLR_RED}${CLR_BOLD}[ERR]${CLR_RESET}  %s\n" "$*" >&2; }

# ------------------------------------------------------------------------------
# 1. Configuration (Environment overrides supported)
# ------------------------------------------------------------------------------
DEFAULT_SERVER_BIN="/run/media/rakuya/Data/Native Llama/bin/b11838-native/bin/llama-server"
DEFAULT_MODEL_PATH="/run/media/rakuya/Data/Native Llama/llama.cpp/build-linux/bin/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"

SERVER_BIN="${RENVI_SERVER_BIN:-$DEFAULT_SERVER_BIN}"
MODEL_PATH="${RENVI_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
HOST="${RENVI_HOST:-127.0.0.1}"
PORT="${RENVI_PORT:-8080}"

CTX_SIZE="${RENVI_CTX:-6144}"
SLOTS="${RENVI_SLOTS:-1}"
THREADS="${RENVI_THREADS:-4}"                # 4 physical cores saturate dual-channel DDR5 without thermal throttling
BATCH="${RENVI_BATCH:-1024}"
UBATCH="${RENVI_UBATCH:-512}"
VRAM_MARGIN_MIB="${RENVI_VRAM_MARGIN:-512}"  # Reserve headroom for KDE/Wayland compositor
MAX_TOKENS="${RENVI_MAX_TOKENS:-3500}"
PROMPT_CACHE_MIB="${RENVI_PROMPT_CACHE_MIB:-512}"
CTX_CHECKPOINTS="${RENVI_CTX_CHECKPOINTS:-4}"
TEMP="${RENVI_TEMP:-0.3}"
DEVICE="${RENVI_DEVICE:-}"
WEBUI="${RENVI_WEBUI:-0}"
SLEEP_IDLE="${RENVI_SLEEP_IDLE:--1}"
WAIT_SECS="${RENVI_WAIT_SECS:-240}"
LOAD_MODE="${RENVI_LOAD_MODE:-mmap}"

# ------------------------------------------------------------------------------
# 2. Pre-Flight Diagnostics
# ------------------------------------------------------------------------------
printf "${CLR_BOLD}${CLR_CYAN}"
echo "================================================================================"
echo "          Renvi / ViTrans — Native MoE LLM Server (llama.cpp)                   "
echo "================================================================================"
printf "${CLR_RESET}"

[[ -x "$SERVER_BIN" ]] || { msg_err "Executable not found: $SERVER_BIN"; exit 1; }
msg_ok "Binary verified: $SERVER_BIN"

[[ -f "$MODEL_PATH" ]] || { msg_err "Model file not found: $MODEL_PATH"; exit 1; }
MODEL_BYTES=$(stat -c %s "$MODEL_PATH")
MODEL_MIB=$(( MODEL_BYTES / 1024 / 1024 ))
msg_ok "Model verified : $(basename "$MODEL_PATH") (${MODEL_MIB} MiB)"

if curl -s -o /dev/null --max-time 1 "http://${HOST}:${PORT}/health"; then
    msg_err "Port ${PORT} is already in use by an active server."
    exit 1
fi

# Hardware discovery
FREE_VRAM_MIB=0
TOTAL_VRAM_MIB=0
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    TOTAL_VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    FREE_VRAM_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    msg_ok "GPU detected   : ${GPU_NAME} [VRAM: ${FREE_VRAM_MIB} MiB free / ${TOTAL_VRAM_MIB} MiB total]"
else
    msg_warn "nvidia-smi not available; proceeding with default device discovery."
fi

AVAIL_RAM_MIB=$(free -m | awk '/^Mem:/{print $7}')
TOTAL_RAM_MIB=$(free -m | awk '/^Mem:/{print $2}')
msg_ok "RAM status     : ${AVAIL_RAM_MIB} MiB available / ${TOTAL_RAM_MIB} MiB total"

# Zero-SSD wear verification
GPU_USABLE=$(( FREE_VRAM_MIB - VRAM_MARGIN_MIB - 700 )); (( GPU_USABLE < 0 )) && GPU_USABLE=0
CPU_PART_MIB=$(( MODEL_MIB - GPU_USABLE )); (( CPU_PART_MIB < 0 )) && CPU_PART_MIB=0
msg_info "Estimated RAM weights: ~${CPU_PART_MIB} MiB (resident memory, zero swap/SSD paging)"

# ------------------------------------------------------------------------------
# 3. CPU Core Pinning (Physical Cores Only, BORE Kernel friendly)
# ------------------------------------------------------------------------------
PHYS_CORES=$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)
(( PHYS_CORES > 0 )) || PHYS_CORES=8
(( THREADS > PHYS_CORES )) && THREADS=$PHYS_CORES

picked=0
CPU_MASK=0
picked_cpus=()
declare -A seen_core=()
while IFS=, read -r cpu core; do
    [[ "$cpu" =~ ^[0-9]+$ ]] || continue
    if [[ -z "${seen_core[$core]:-}" ]]; then
        seen_core[$core]=1
        if (( picked < THREADS )); then
            CPU_MASK=$(( CPU_MASK | (1 << cpu) ))
            picked_cpus+=("$cpu")
            picked=$(( picked + 1 ))
        fi
    fi
done < <(lscpu -p=CPU,CORE 2>/dev/null)

# CPU_ARGS=(-t "$THREADS" -tb "$THREADS")
CPU_ARGS=(-t "$THREADS" -tb 12)
if (( CPU_MASK != 0 )); then
    CPU_MASK_HEX=$(printf '%x' "$CPU_MASK")
    CPU_ARGS+=(--cpu-mask "0x${CPU_MASK_HEX}" --cpu-mask-batch "0x${CPU_MASK_HEX}" --cpu-strict 1 --cpu-strict-batch 1)
    picked_cpus_str="${picked_cpus[*]}"
    msg_ok "CPU pinning    : ${THREADS} threads pinned to physical cores [${picked_cpus_str}] (Mask: 0x${CPU_MASK_HEX})"
else
    msg_warn "Could not read CPU core topology; running without core pinning."
fi

# ------------------------------------------------------------------------------
# 4. CUDA Device Selection
# ------------------------------------------------------------------------------
DEV_ARGS=()
if [[ -z "$DEVICE" ]]; then
    DEVICE=$("$SERVER_BIN" --list-devices 2>/dev/null | grep -oE '^[[:space:]]*CUDA[0-9]+' | tr -d '[:space:]' | head -n1 || true)
fi
if [[ -n "$DEVICE" ]]; then
    DEV_ARGS=(-dev "$DEVICE")
    msg_ok "Compute device : ${DEVICE} (NVIDIA Dedicated)"
fi

# ------------------------------------------------------------------------------
# 5. Build Final Execution Command
# ------------------------------------------------------------------------------
CMD=(
    "$SERVER_BIN"
    -m "$MODEL_PATH"
    --host "$HOST" --port "$PORT"
    -c "$CTX_SIZE" -np "$SLOTS"
    -b "$BATCH" -ub "$UBATCH"
    -fa on
    -ctk q8_0 -ctv q8_0
    -fit on -fitt "$VRAM_MARGIN_MIB"
    -lm "$LOAD_MODE" -lzm off
    "${CPU_ARGS[@]}"
    --poll 0
    -cram "$PROMPT_CACHE_MIB"
    -ctxcp "$CTX_CHECKPOINTS"
    # --no-warmup
    --reasoning off
    --no-reasoning-preserve
    --reasoning-format none
    --reasoning-budget 0
    --skip-chat-parsing
    -n "$MAX_TOKENS"
    --temp "$TEMP" --top-k 20 --top-p 0.8
    --sleep-idle-seconds "$SLEEP_IDLE"
)
(( ${#DEV_ARGS[@]} )) && CMD+=("${DEV_ARGS[@]}")
[[ "$WEBUI" == "1" ]] || CMD+=(--no-webui)

printf "\n${CLR_CYAN}${CLR_BOLD}[LAUNCH COMMAND]${CLR_RESET}\n"
printf "  ${CLR_GRAY}%s${CLR_RESET}\n\n" "${CMD[*]}"

# ------------------------------------------------------------------------------
# 6. Process Execution & Health Monitoring
# ------------------------------------------------------------------------------
SERVER_PID=""
cleanup() {
    trap - INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        printf "\n${CLR_YELLOW}${CLR_BOLD}[SHUTDOWN] Đang dừng llama-server an toàn (PID: %s)...${CLR_RESET}\n" "$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        msg_ok "llama-server đã dừng."
    fi
    exit 0
}
trap cleanup INT TERM

"${CMD[@]}" &
SERVER_PID=$!

msg_info "Tiến trình llama-server khởi động (PID: ${SERVER_PID}). Đang tải model vào VRAM & RAM..."
printf "${CLR_GRAY}────────────────────────────────────────────────────────────────────────────────${CLR_RESET}\n"

HEALTH_OK=0
for (( i = 0; i < WAIT_SECS / 2; i++ )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        rc=0; wait "$SERVER_PID" 2>/dev/null || rc=$?
        printf "${CLR_GRAY}────────────────────────────────────────────────────────────────────────────────${CLR_RESET}\n"
        msg_err "Tiến trình máy chủ thoát sớm (exit code ${rc})."
        if (( rc == 137 )); then
            msg_err "Nguyên nhân: Bị Linux OOM Killer tắt do thiếu RAM. Hãy đóng bớt ứng dụng."
        fi
        exit 1
    fi
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://${HOST}:${PORT}/health" 2>/dev/null || true)
    if [[ "$code" == "200" ]]; then
        HEALTH_OK=1
        break
    fi
    sleep 2
done

printf "${CLR_GRAY}────────────────────────────────────────────────────────────────────────────────${CLR_RESET}\n"

if (( HEALTH_OK )); then
    printf "${CLR_GREEN}${CLR_BOLD}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    ✔ MÁY CHỦ LLM ĐÃ TRỰC TUYẾN & SẴN SÀNG                   ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    printf "║ • Endpoint API   : http://%s:%-49s║\n" "${HOST}" "${PORT}/v1"
    printf "║ • Kiểm tra sức khỏe: http://%s:%-47s║\n" "${HOST}" "${PORT}/health"
    printf "║ • Model kích hoạt: %-58s║\n" "$(basename "$MODEL_PATH")"
    printf "║ • Cơ chế Offload : Auto-fit (-fit on, VRAM margin %d MiB, KV: q8_0)     ║\n" "$VRAM_MARGIN_MIB"
    printf "║ • CPU & Nhiệt độ : %d nhân vật lý [%s] | --poll 0 (Không tốn CPU chờ)  ║\n" "$THREADS" "${picked_cpus_str:-auto}"
    printf "║ • Tuổi thọ SSD   : An toàn tuyệt đối (-lzm off, -lm mmap, 0 ghi đĩa)        ║\n"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    printf "${CLR_RESET}\n"
    msg_ok "Hệ thống sẵn sàng phục vụ tiến trình dịch ViTrans."
    msg_info "Để chạy dịch tự động trong terminal khác:"
    printf "  ${CLR_CYAN}python scripts/unattended_runner.py translate -p \"<đường-dẫn-game>\"${CLR_RESET}\n"
    msg_info "Nhấn ${CLR_BOLD}Ctrl+C${CLR_RESET} để dừng máy chủ bất cứ lúc nào.\n"
else
    msg_warn "Hết thời gian chờ kiểm tra /health. Máy chủ có thể vẫn đang tải weights lớn..."
fi

wait "$SERVER_PID"
