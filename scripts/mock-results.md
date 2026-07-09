# opencode CLI latency — mock LLM (stub-anthropic/stub-1)

## 4 matched cases — exact tool-call sequence/step count replayed from the real run

Same case names, prompts, and tool-call sequence as real-results.md's matched section. Only the model round-trip is faked (real tool execution still happens).

| phase | 01-pure-text | 03-bash | 08-bash-chain | 10-complex |
|---|---|---|---|---|
| llm | 99.2ms (31.3%) | 334.0ms (55.6%) | 813.5ms (62.6%) | 712.4ms (61.9%) |
|   ↳ tool | 0.0ms (0.0%) | 79.1ms (13.2%) | 275.6ms (21.2%) | 196.4ms (17.1%) |
| non_llm_setup | 84.4ms (26.6%) | 99.8ms (16.6%) | 142.5ms (11.0%) | 126.8ms (11.0%) |
| session_load | 3.3ms (1.0%) | 4.4ms (0.7%) | 7.5ms (0.6%) | 7.2ms (0.6%) |
| get_model | 0.4ms (0.1%) | 0.5ms (0.1%) | 0.7ms (0.1%) | 0.8ms (0.1%) |
| reminders | 0.7ms (0.2%) | 0.7ms (0.1%) | 0.9ms (0.1%) | 0.9ms (0.1%) |
| msg_persist | 2.1ms (0.6%) | 3.7ms (0.6%) | 8.0ms (0.6%) | 8.8ms (0.8%) |
| processor_create | 119.3ms (37.6%) | 146.2ms (24.3%) | 303.7ms (23.4%) | 272.0ms (23.6%) |
|   ↳ snapshot_track | 118.7ms (37.4%) | 145.4ms (24.2%) | 302.5ms (23.3%) | 270.8ms (23.5%) |
|     ↳ snapshot_add | 111.8ms (35.3%) | 129.4ms (21.5%) | 311.5ms (24.0%) | 298.8ms (26.0%) |
|     ↳ snapshot_write_tree | 8.1ms (2.6%) | 36.6ms (6.1%) | 67.7ms (5.2%) | 63.6ms (5.5%) |
| finalize | 0.1ms (0.0%) | 0.2ms (0.0%) | 0.3ms (0.0%) | 0.2ms (0.0%) |
| background (concurrent, not in total) | 102.9ms | 118.2ms | 117.0ms | 113.2ms |
| **unaccounted** | **7.7ms (2.4%)** | **11.8ms (2.0%)** | **21.4ms (1.7%)** | **21.8ms (1.9%)** |
| **total_wall_ms** | **317.0ms** | **601.2ms** | **1298.6ms** | **1150.9ms** |
| steps | 2 | 3 | 6 | 6 |

## 4 generic cases (0/1/3/10 tool calls) — synthetic prompts, not paired with real runs

| phase | stub-00-notool | stub-01-tool | stub-03-chain | stub-10-chain |
|---|---|---|---|---|
| llm | 104.3ms (30.0%) | 272.5ms (47.4%) | 554.7ms (58.2%) | 1416.3ms (63.0%) |
|   ↳ tool | 0.0ms (0.0%) | 37.4ms (6.5%) | 128.0ms (13.4%) | 325.8ms (14.5%) |
| non_llm_setup | 91.0ms (26.1%) | 91.0ms (15.8%) | 118.8ms (12.5%) | 222.8ms (9.9%) |
| session_load | 3.6ms (1.0%) | 4.6ms (0.8%) | 6.3ms (0.7%) | 13.2ms (0.6%) |
| get_model | 0.4ms (0.1%) | 0.5ms (0.1%) | 0.7ms (0.1%) | 1.4ms (0.1%) |
| reminders | 0.7ms (0.2%) | 0.7ms (0.1%) | 0.8ms (0.1%) | 1.3ms (0.1%) |
| msg_persist | 2.2ms (0.6%) | 3.7ms (0.6%) | 6.1ms (0.6%) | 15.9ms (0.7%) |
| processor_create | 137.6ms (39.6%) | 190.6ms (33.1%) | 248.5ms (26.1%) | 533.6ms (23.7%) |
|   ↳ snapshot_track | 136.9ms (39.4%) | 189.8ms (33.0%) | 247.5ms (26.0%) | 531.5ms (23.6%) |
|     ↳ snapshot_add | 125.5ms (36.1%) | 167.9ms (29.2%) | 225.7ms (23.7%) | 539.7ms (24.0%) |
|     ↳ snapshot_write_tree | 9.2ms (2.6%) | 24.4ms (4.2%) | 38.5ms (4.0%) | 128.9ms (5.7%) |
| finalize | 0.1ms (0.0%) | 0.1ms (0.0%) | 0.3ms (0.0%) | 0.6ms (0.0%) |
| background (concurrent, not in total) | 115.5ms | 122.4ms | 114.8ms | 113.1ms |
| **unaccounted** | **8.0ms (2.3%)** | **11.8ms (2.0%)** | **16.8ms (1.8%)** | **42.2ms (1.9%)** |
| **total_wall_ms** | **347.8ms** | **575.4ms** | **953.0ms** | **2247.6ms** |
| steps | 2 | 3 | 5 | 12 |
