#!/bin/bash -e

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

results_dir=$1

pushd $results_dir

echo "tools/train.py CPU %:"
grep tools/train.py top.log | awk '{ sum += $9} END { if (NR > 0) printf ("%0.3f\n", sum / NR)}'

echo "Memory Read, Write, IO (GB/s):"
grep ^MEM -A2 pcm.log | grep SKT | awk '{print $3}' | awk '{sum += $1 } END { if (NR > 0) printf ("%0.3f, ", sum / NR)}'
grep ^MEM -A2 pcm.log | grep SKT | awk '{print $4}' | awk '{sum += $1 } END { if (NR > 0) printf ("%0.3f, ", sum / NR)}'
grep ^MEM -A2 pcm.log | grep SKT | awk '{print $5}' | awk '{sum += $1 } END { if (NR > 0) printf ("%0.3f\n", sum / NR)}'


echo "Total CPU%:"
grep "top -" top.log -A 16 | grep -v top | grep -v Tasks | grep -v "%Cpu(s)" | grep -v KiB | grep -v PID | awk '{ printf "%s"",",$9 }' | sed -e 's/,,,/\n/g' | awk '{ for(i=1; i<=NF; i++) j+=$i; print j; j=0 }' | awk '{ sum +=$1 } END { if (NR > 0) print sum / NR}'
echo "Top Average Percent Memory usage:"
avg=`grep 'KiB Mem' top.log | awk '{ sum += $8 } END { if (NR > 0) printf ("%0.3f\n", sum / NR)}'`
total=`grep 'KiB Mem' top.log | awk '{ sum += $4 } END { if (NR > 0) printf ("%0.3f\n", sum / NR)}'`
echo "scale=4; $avg / $total" | bc
echo "Top Memory utilization (KiB):"
echo "$avg"
popd
