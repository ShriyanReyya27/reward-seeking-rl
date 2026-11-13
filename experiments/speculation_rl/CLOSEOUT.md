# Stage 2 continuation

Use the local results directory phase1-screen/results/speculation_s1_20261112. All sampling decisions and seeds are recorded in the per-request files. The spend ceiling uses the shared campaign ledger. Inspect current process command lines before starting any runner. Current jobs may continue independently of the chat UI.

- screen.py is running the full frozen pool. The opening run handled indices 0 and 1. The resumed worker handles indices 2–49 with independent family queues. Existing completed groups and raw turns are reused.
- frontier_block.py is running the fresh A/B/C comparison, indices 2–15. The separate opening sample and the GLM engineering smoke are complete.
- continue_screen.py confirmation-labels waits for the paired identity samples, then labels available traces with both judges.
- continue_screen.py pilots waits for all non-missing S1 groups, labels the prespecified sample, then runs P1 followed by P2. Its waiting code recognises documented unresolved requests.
- step0_battery and step0_coding finished. The branch contains native metrics, qualitative limits, transcripts and missingness in stage2_diagnostics.

Check each process and the actual result files if the workstation sleeps. After an error compare reservations with saved raw replies. Settle recoverable replies from their recorded usage. Preserve ambiguously dispatched reservations and missingness. Never charge a second time for the same ambiguous request.

The completed opening and diagnostic results are published. To close the stage, use report_screen.py on all 50 indices, gates.py on its table and the completed screen_labels folder, analyze_framing.py on framing_gradient, and analyze_confirmation.py on confirmation_labels. Publish verified raw records and group-level and task-paired tables on the existing branch. Any keep decision must also address conditional correct/incorrect speculation rates, independent wording review deferral, and both judges. Review the evidence before dispatching a scientific RL run. The disposable full GLM adapter must be excluded from every comparison.
