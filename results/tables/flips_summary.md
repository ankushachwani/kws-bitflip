Flips to drive 12-class evaluation accuracy to <= 15% (chance = 8.3%). Mean over the seeds that reached it; "reached" counts seeds (of 3) that hit the target within 30 flips. "flips/Mbit" = mean flips per million stored weight bits.

| model | variant | base acc | PBS flips [min,max] (reached) | PBS flips/Mbit | random flips (reached) | random final acc |
|---|---|---|---|---|---|---|
| DS-CNN-S | int8 (8-bit W) | 0.947 | 5.3 [5,6] (3/3) | 30.281 | not reached (0/3) | 0.941 |
| DS-CNN-S | w4a8 (4-bit W) | 0.923 | 5.7 [5,7] (3/3) | 64.347 | not reached (0/3) | 0.913 |
| DS-CNN-S | w4a4 (4-bit W) | 0.893 | 8.0 [6,9] (3/3) | 90.843 | not reached (0/3) | 0.881 |
| TC-ResNet8 | int8 (8-bit W) | 0.953 | not reached (0/3) | - | not reached (0/3) | 0.950 |
| TC-ResNet8 | w4a8 (4-bit W) | 0.941 | not reached (0/3) | - | not reached (0/3) | 0.944 |
| TC-ResNet8 | w4a4 (4-bit W) | 0.931 | not reached (0/3) | - | not reached (0/3) | 0.928 |
