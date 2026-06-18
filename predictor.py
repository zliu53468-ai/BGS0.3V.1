Baccarat V9.2 Local Simulation Summary
Seed: 20260618
Test rounds: 1000 prediction opportunities
Settlement rule: Banker/Player recommendations only; Tie result treated as push and excluded from win/loss winrate.

Candidate results from first 1000-round scan:
- A_平衡低門檻: 100W / 109L / 25P, settled 209, actions 234, winrate 47.85%, action 23.4%, observe 76.6%, max losing 7
- B_中等準度: 56W / 53L / 11P, settled 109, actions 120, winrate 51.38%, action 12.0%, observe 88.0%, max losing 4
- C_補牌重中門檻: 78W / 80L / 20P, settled 158, actions 178, winrate 49.37%, action 17.8%, observe 82.2%, max losing 8
- D_高出手: 157W / 159L / 35P, settled 316, actions 351, winrate 49.68%, action 35.1%, observe 64.9%, max losing 6
- E_保守: 2W / 5L / 1P, settled 7, actions 8, winrate 28.57%, action 0.8%, observe 99.2%, max losing 2
- F_點數主: 71W / 74L / 17P, settled 145, actions 162, winrate 48.97%, action 16.2%, observe 83.8%, max losing 6
- G_combo補牌: 123W / 136L / 29P, settled 259, actions 288, winrate 47.49%, action 28.8%, observe 71.2%, max losing 8

Final formal run for recommended B with MC=1500 and composition=1500:
- B_中等準度_正式MC1500: 84W / 82L / 21P, settled 166, actions 187, winrate 50.60%, action 18.7%, observe 81.3%, max losing 6

Recommended practical parameter style:
Use B_中等準度 as base. If you want slightly more actions, use MC=1500/COMPOSITION=1500 formal variant; if you want highest one-run winrate, use B initial scan thresholds with MC=800.
