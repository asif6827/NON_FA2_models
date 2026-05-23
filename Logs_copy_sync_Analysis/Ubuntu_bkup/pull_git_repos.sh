set -euo pipefail

# Optional: set a base directory where these repos live
# Example: BASE_DIR="/export/home/asifali"
BASE_DIR="$(pwd)"

echo "Base dir: $BASE_DIR"
echo "Updating repos..."
echo 



# ---------- Reasoning360_sys_B1 ----------
echo "Pulling Reasoning360_sys_B1"
cd "$BASE_DIR/Reasoning360_sys_B1" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B1"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B1"; exit 1; }
git pull --ff-only
cd ..
echo 


# ---------- Reasoning360_sys_B3 ----------
echo "Pulling Reasoning360_sys_B3"
cd "$BASE_DIR/Reasoning360_sys_B3" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B3"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B3"; exit 1; }
git pull --ff-only
cd ..
echo 



# ---------- Reasoning360_sys_B_v3 ----------
echo "Pulling Reasoning360_sys_B_v3"
cd "$BASE_DIR/Reasoning360_sys_B_v3" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B_v3"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B_v3"; exit 1; }
git pull --ff-only
cd ..
echo 


# ---------- Reasoning360_sys_B_v4 ----------
echo "Pulling Reasoning360_sys_B_v4"
cd "$BASE_DIR/Reasoning360_sys_B_v4" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B_v4"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B_v4"; exit 1; }
git pull --ff-only
cd ..
echo 


# ---------- Reasoning360 ----------
echo "Pulling Reasoning360"
cd "$BASE_DIR/Reasoning360" || { echo "Missing dir: $BASE_DIR/Reasoning360"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360"; exit 1; }
git pull --ff-only
cd ..
echo 

# ---------- Reasoning360_NL ----------
echo "Pulling Reasoning360_NL"
cd "$BASE_DIR/Reasoning360_NL" || { echo "Missing dir: $BASE_DIR/Reasoning360_NL"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_NL"; exit 1; }
git pull --ff-only
cd ..
echo 

# ---------- Reasoning360_sys_A ----------
echo "Pulling Reasoning360_sys_A"
cd "$BASE_DIR/Reasoning360_sys_A" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_A"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_A"; exit 1; }
git pull --ff-only
cd ..
echo 

# ---------- Reasoning360_sys_B ----------
echo "Pulling Reasoning360_sys_B"
cd "$BASE_DIR/Reasoning360_sys_B" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B"; exit 1; }
git pull --ff-only
cd ..
echo 


# ---------- Reasoning360_sys_B-Parsed-V3----------
echo "Pulling Reasoning360_sys_B_Parsed_V3"
cd "$BASE_DIR/Reasoning360_sys_B_Parsed_V3" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B_Parsed_V3"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B_Parsed_V3"; exit 1; }
git pull --ff-only
cd ..
echo 



# ---------- Reasoning360_parsed_Asif ----------
echo "Pulling Reasoning360_parsed_Asif"
cd "$BASE_DIR/Reasoning360_parsed_Asif" || { echo "Missing dir: $BASE_DIR/Reasoning360_parsed_Asif"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_parsed_Asif"; exit 1; }
git pull --ff-only
cd ..
echo 

# ---------- Reasoning360_sys_B_v1 ----------
echo "Pulling Reasoning360_sys_B_v1"
cd "$BASE_DIR/Reasoning360_sys_B_v1" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B_v1"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B_v1"; exit 1; }
git pull --ff-only
cd ..
echo 


# ---------- Reasoning360_sys_B_v2 ----------
echo "Pulling Reasoning360_sys_B_v2"
cd "$BASE_DIR/Reasoning360_sys_B_v2" || { echo "Missing dir: $BASE_DIR/Reasoning360_sys_B_v2"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo: Reasoning360_sys_B_v2"; exit 1; }
git pull --ff-only
cd ..
echo 


