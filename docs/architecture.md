# Architecture

Everything inside the dashed box runs in the s390x container on IBM LinuxONE. Data never enters the repo.

```mermaid
flowchart LR
  classDef data stroke:#d29922
  classDef model stroke:#58a6ff
  classDef alert stroke:#f85149

  G["<b>Game data</b><br/>opencs2 pro demos · demoparser2<br/>1,988 maps · 42,009 rounds"]:::data
  M["<b>Market data</b><br/>Polymarket public APIs<br/>12,489 events · 1-min prices · every fill"]:::data

  subgraph Z["IBM LinuxONE · s390x · Python 3.12 · no GPU"]
    direction LR
    J["<b>Join + align</b><br/>one clock, 64-tick<br/>timezone fixed per map<br/>1,284 → 1,014 maps<br/>pipeline/io.py · align.py"]
    A["<b>A · Game model</b><br/>P(win) every round<br/>logistic regression<br/>Brier 0.178 (coin flip 0.25)<br/>models/train.py"]:::model
    B["<b>B · Market price</b><br/>map-winner probability<br/>pipeline/market.py"]:::model
    CD["<b>C · D · Supporting signals</b><br/>player off own baseline<br/>price steps · trade size<br/>pipeline/detect.py"]
    E["<b>E · Alert rule</b><br/>gap ≥ 15 pts for 2+ rounds<br/>market moved first<br/>no game event explains it<br/>pipeline/detect.py"]:::alert
    S["<b>Dashboard + API</b><br/>alert with exact top factors<br/>~9 ms per score<br/>server/app.py · site/"]
  end

  G --> J
  M --> J
  J --> A
  J --> B
  J --> CD
  A --> E
  B --> E
  CD -.-> E
  E --> S
  S --> H(["Human reviewer decides"])
```

F (`pipeline/inject.py`) inserts labelled, simulated prints for the demo alert; it is left out of the diagram for clarity.

Team One 95 · IBM Z Datathon 2026 · SSF2026
