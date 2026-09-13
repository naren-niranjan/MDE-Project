# Acquisition rate measurement

Window 60.3 s, 4 cameras, RGB8 at 2448 x 2048.

| quantity | value |
|---|---|
| complete frame sets | 223 |
| mean rate | 3.70 sets/s |
| median instantaneous | 3.70 sets/s |
| 5th--95th percentile | 3.70--3.71 sets/s |
| incomplete sets | 0 |
| per-camera frames | 223, 223, 223, 223 |
| bytes per frame | 15.04 MB |
| bytes per set | 60.16 MB |
| payload moved | 222.4 MB/s |
| payload bound at 250 MB/s | 4.16 sets/s |
| shortfall against bound | 11.1 % |

## Sentence for Section 5.5

Sustained acquisition was measured over 60\,s of continuous
streaming: 223 complete four-camera sets, a mean of
3.70\,sets per second with a median instantaneous rate of
3.70 and a 5th-to-95th percentile band of 3.70 to 3.71.
At 60.16\,MB per synchronised set this moves 222\,MB/s,
against a payload bound of 4.16\,sets per second on the
250\,MB/s budget --- a shortfall of 11\,\%, attributable to
protocol and scheduling overhead not separated here.
