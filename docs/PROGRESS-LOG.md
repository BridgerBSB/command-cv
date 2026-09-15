# Command CV - running progress log

One line per thing. Newest at the top. Status is DONE / OPEN / WRONG (retracted).
Visuals for each step go to `C:\Users\Owner\Desktop\command-cv-progress\`.

| # | Date | What | Status | Evidence |
|---|---|---|---|---|
| 17 | 09-15 | Training set is **MLB ONLY** - 3 parks (Camden/Detroit/Colorado), 15,900 crops, ZERO A-ball or AA. The held-out val split was still an MLB park | DONE | `06_scale_gap.png` |
| 16 | 09-15 | SCALE THEORY RETRACTED. Training balls 8.5-12.3 px; cu_2's ball ~11.7 px (21 px box / 1.8), dead in range. ch_1's framing is comparable to cu_2, so its ball is in range too. I read "tighter camera" off a crop I had upscaled 3x myself | WRONG->fixed | `06_scale_gap.png` |
| 15 | 09-14 | ~~A+ feeds differ a lot park to park; ch_1 is a tight zoom~~ framings are actually comparable | WRONG->fixed | `06_scale_gap.png` |
| 14 | 09-14 | `ch_1`: ball IS visible, detector fires 6-10x/frame ALL on the catcher's shin guard, never the ball. Cause UNKNOWN. Candidate: harsh direct sun + hard shadows vs flat overcast in training. NOT verified | OPEN | `05_ch1_corridor.png` |
| 13 | 09-14 | PITCH = FIRST ball flight in the clip (Zac's rule, camera-independent). Tracker now picks f188-204 on the whole cu_2 clip; 5 distinct events listed in time order | DONE | `04_events_pitch_picked.png` |
| 12 | 09-14 | ~~Separate pitch from throw by direction~~ superseded by #13 - first-flight is simpler and needs no per-camera knowledge | DONE | |
| 11 | 09-14 | `ogando_ch_1` yields 0 tracks over its whole clip | EXPLAINED by #14 | |
| 10 | 09-14 | RETRACTED "camera pan carried a wall light". Camera is static (0.15 px/fr median). It is a REAL second baseball = the catcher's return throw | WRONG->fixed | `1be409dd` |
| 9 | 09-14 | Tracker v4: top-K chains, break on direction reversal, score sub-chains. Recovers the known pitch f188-204 ranked #1 in a flight window; picks the return throw over the whole clip | DONE (partial) | scratchpad, not promoted |
| 8 | 09-13 | Tracker v1/v2/v3 all failed validation, each for a different reason | DONE (recorded) | `e29a03c8` |
| 7 | 09-13 | A+ TRANSFERS: detector tracks the pitch in the unseen Asheville feed, conf to 0.564, baseball with seams | DONE | `9ae1c003`, `output/aplus_zoom_cu2.png` |
| 6 | 09-13 | Three A+ search criteria failed; a stride-10 contact strip found the delivery BY EYE in one render | DONE | `a2783c29` |
| 5 | 09-13 | A+ manifests DO carry a 9-param trajectory. A+ lacks the drawn zone box, not the trajectory | DONE | `a2783c29` |
| 4 | 09-13 | REAL GATE PASSES: camera from our detections 0.06/0.15 ft off box-assisted, held-out corners 0.98/1.01 in (59 clips) | DONE | `5173a2f1` |
| 3 | 09-13 | Our labels match OpenCommand's independent detections to 2.15 px (0.65 in), 277/277 plays under 10 px | DONE | `5cc66914` |
| 2 | 09-13 | Detector is fine: 97.7% found @ 3.8-4.5 px on the same weights (was 29-48% @ 167-308 px) | DONE | `5cc66914` |
| 1 | 09-13 | FOUND THE BUG: the gate read frames with no ball in them (OC frame_idx counts from RELEASE, ours from clip start) | DONE | `5cc66914` |

## Things I got wrong and corrected myself

| What I claimed | Why it was wrong | Where |
|---|---|---|
| "The detector is undertrained, needs more epochs" | It was being scored on pre-pitch frames | `5cc66914` |
| "The 18-frame run is a wall light carried by a camera pan" | Camera measures 0.15 px/fr. It is a real second baseball | `1be409dd` |
| "A+ ball is a motion-blurred transverse streak" | It is a compact round blob, same as training data | `9ae1c003` |

## Still needs Zac

- MEMORY.md is 19.6KB vs a 17.1KB target. Cutting further means deleting entries I did not write.
- The 09-12 Command CV LINEAGE entry sits at the file tail, not its date position.
- Training stays stopped at epoch 16/30 unless you say otherwise. "Needs more epochs" is void.
