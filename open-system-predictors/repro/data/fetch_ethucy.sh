#!/usr/bin/env bash
# Fetch the five ETH/UCY pedestrian-trajectory scenes used by experiment E1 and check them against the files used.
#
# The ETH (BIWI Walking Pedestrians) and UCY (Crowds-by-Example) data are not redistributed with this supplement:
# the ETH terms allow research use only and grant no redistribution right, and the UCY terms ask for credit to the
# original uploaders. The files are fetched from the OpenTraj repository, pinned to the commit whose files are
# byte-identical to the ones used here (checked by git blob hash on 2026-10-30).
#
# Please cite: S. Pellegrini, A. Ess, K. Schindler, L. Van Gool, "You'll never walk alone: modeling social behavior
# for multi-target tracking", ICCV 2009 (ETH); A. Lerner, Y. Chrysanthou, D. Lischinski, "Crowds by Example",
# Computer Graphics Forum 26(3), 2007 (UCY); and OpenTraj (J. Amirian et al.) as the packaging source.
#
# Output: data/ethucy/raw/{eth,hotel,univ,zara01,zara02}_obsmat.txt (the path src/osw/tasks_ethucy.py reads).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="${HERE}/ethucy/raw"
COMMIT="97ec1d5e1b579f26febeaa9083705952deac338a"
BASE="https://raw.githubusercontent.com/crowdbotp/OpenTraj/${COMMIT}/datasets"
mkdir -p "${RAW}"

fetch() {  # scene, OpenTraj folder, expected SHA-256
  local scene="$1" src="$2" want="$3" out="${RAW}/$1_obsmat.txt"
  curl -fsSL -m 180 -o "${out}" "${BASE}/${src}/obsmat.txt"
  local got
  got="$(shasum -a 256 "${out}" | cut -d' ' -f1)"
  if [ "${got}" != "${want}" ]; then
    echo "${scene}: SHA-256 mismatch (got ${got}, expected ${want})" >&2
    exit 1
  fi
  echo "${scene}: ok"
}

fetch eth    ETH/seq_eth     d452ae2185ecb1164c2fdf31e75f6236f4c2ffc02c751a6b2ae921740cbc60d1
fetch hotel  ETH/seq_hotel   2b8577595204a6b780b80258fd50da35adb7293e3e06ad6f8e1b60965e7dde6e
fetch univ   UCY/students03  d9fd7048eab7d3d9550d7ac4f59b19d97a87612310e8c8e48cf3aaf59764ff80
fetch zara01 UCY/zara01      bd999ebcfbc506821fa5032a10b9101e150defcff85f1ef037d09ce7c6a8c2aa
fetch zara02 UCY/zara02      0b29bdacd54b971299b741f291d9b0eac0ce9aeec9901e75576c529a39523e9b
