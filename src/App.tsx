import { useState, useEffect, useRef } from "react";

const C = {
  bg: "#080b12",
  panel: "#0d1120",
  border: "#1a2035",
  borderHi: "#2a3555",
  text: "#e2e8f8",
  muted: "#4a5578",
  dim: "#2a3050",
  green: "#00e5a0",
  red: "#ff4466",
  amber: "#f5a623",
  blue: "#3d8ef8",
  cyan: "#00c8e0",
  purple: "#8b5cf6",
  greenDim: "#00e5a015",
  redDim: "#ff446615",
  amberDim: "#f5a62315",
};

function seedRand(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff;
    return (s >>> 0) / 0xffffffff;
  };
}

interface PricePoint {
  day: number;
  open: number;
  close: number;
  high: number;
  low: number;
  price: number;
  volume: number;
  ret: number;
  vol?: number | null;
  volPct?: number | null;
  rsi?: number;
  ma7?: number;
  ma20?: number;
  priceMa7Ratio?: number;
  priceMa20Ratio?: number;
  volRank?: number;
}

function generatePriceHistory(coin: string, days = 90): PricePoint[] {
  const rand = seedRand(coin.charCodeAt(0) * 137 + coin.charCodeAt(1) * 31);
  const prices: PricePoint[] = [];
  let price: number =
    ({ BTC: 42000, ETH: 2200, BNB: 310, SOL: 95, ADA: 0.45 } as Record<string, number>)[coin] ?? 100;
  for (let i = 0; i < days; i++) {
    const vol = 0.02 + rand() * 0.04;
    const ret = (rand() - 0.48) * vol * 2;
    price *= 1 + ret;
    const open = price;
    const close = price * (1 + (rand() - 0.5) * 0.015);
    const high = Math.max(open, close) * (1 + rand() * 0.01);
    const low = Math.min(open, close) * (1 - rand() * 0.01);
    const volume = rand() * 8e9 + 1e9;
    prices.push({ day: i, open, close, high, low, price: close, volume, ret });
  }
  return prices;
}

function calcVolatility(prices: PricePoint[], window = 20): PricePoint[] {
  return prices.map((p, i) => {
    if (i < window) return { ...p, vol: null, volPct: null };
    const slice = prices.slice(i - window, i);
    const rets = slice.map((x) => x.ret);
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const std = Math.sqrt(
      rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length
    );
    const annualVol = std * Math.sqrt(252) * 100;
    return { ...p, vol: annualVol, volPct: std * 100 };
  });
}

function calcRSI(prices: PricePoint[], period = 14): PricePoint[] {
  return prices.map((p, i) => {
    if (i < period + 1) return { ...p, rsi: 50 };
    const slice = prices.slice(i - period, i);
    const gains =
      slice.filter((x) => x.ret > 0).reduce((a, b) => a + b.ret, 0) / period;
    const losses =
      slice
        .filter((x) => x.ret < 0)
        .reduce((a, b) => a + Math.abs(b.ret), 0) / period;
    const rs = losses === 0 ? 100 : gains / losses;
    return { ...p, rsi: 100 - 100 / (1 + rs) };
  });
}

function calcFeatures(prices: PricePoint[]): PricePoint[] {
  const withVol = calcVolatility(prices);
  const withRSI = calcRSI(withVol);
  return withRSI.map((p, i) => {
    const ma7 =
      i >= 7
        ? prices.slice(i - 7, i).reduce((a, b) => a + b.price, 0) / 7
        : p.price;
    const ma20 =
      i >= 20
        ? prices.slice(i - 20, i).reduce((a, b) => a + b.price, 0) / 20
        : p.price;
    const priceMa7Ratio = p.price / ma7 - 1;
    const priceMa20Ratio = p.price / ma20 - 1;
    const volRank = p.vol ? Math.min(100, p.vol) / 100 : 0.5;
    return { ...p, ma7, ma20, priceMa7Ratio, priceMa20Ratio, volRank };
  });
}

interface MLModel {
  accuracy: number;
  classes: string[];
  predict: (features: number[]) => { label: string; probs: number[]; confidence: number };
}

function trainVolatilityModel(data: PricePoint[]): MLModel {
  const valid = data.filter((d) => d.vol !== null && d.rsi !== null);
  const X = valid.map((d) => [
    d.volRank ?? 0,
    (d.rsi ?? 50) / 100,
    d.priceMa7Ratio ?? 0,
    d.priceMa20Ratio ?? 0,
    d.ret,
  ]);
  const y = valid.map((d) => ((d.vol ?? 0) > 50 ? 2 : (d.vol ?? 0) > 30 ? 1 : 0));
  const classes = ["Low", "Medium", "High"];
  const weights: number[][] = Array.from({ length: 3 }, () =>
    Array(5)
      .fill(0)
      .map(() => (Math.random() - 0.5) * 0.1)
  );
  const biases = [0, 0, 0];
  function softmax(lg: number[]) {
    const m = Math.max(...lg);
    const e = lg.map((x) => Math.exp(x - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }
  for (let ep = 0; ep < 400; ep++) {
    const dw = weights.map((w) => Array(w.length).fill(0));
    const db = [0, 0, 0];
    for (let i = 0; i < X.length; i++) {
      const lg = weights.map((w, c) =>
        w.reduce((s, wj, j) => s + wj * X[i][j], 0) + biases[c]
      );
      const pr = softmax(lg);
      for (let c = 0; c < 3; c++) {
        const err = pr[c] - (c === y[i] ? 1 : 0);
        for (let j = 0; j < 5; j++) dw[c][j] += err * X[i][j];
        db[c] += err;
      }
    }
    for (let c = 0; c < 3; c++) {
      for (let j = 0; j < 5; j++)
        weights[c][j] -= (0.05 * dw[c][j]) / X.length;
      biases[c] -= (0.05 * db[c]) / X.length;
    }
  }
  let correct = 0;
  X.forEach((x, i) => {
    const lg = weights.map((w, c) =>
      w.reduce((s, wj, j) => s + wj * x[j], 0) + biases[c]
    );
    const pr = softmax(lg);
    if (pr.indexOf(Math.max(...pr)) === y[i]) correct++;
  });
  return {
    accuracy: correct / X.length,
    classes,
    predict(features: number[]) {
      const lg = weights.map((w, c) =>
        w.reduce((s, wj, j) => s + wj * features[j], 0) + biases[c]
      );
      const pr = softmax(lg);
      const idx = pr.indexOf(Math.max(...pr));
      return { label: classes[idx], probs: pr, confidence: Math.max(...pr) };
    },
  };
}

const COINS = ["BTC", "ETH", "BNB", "SOL", "ADA"];
const COIN_NAMES: Record<string, string> = {
  BTC: "Bitcoin",
  ETH: "Ethereum",
  BNB: "BNB Chain",
  SOL: "Solana",
  ADA: "Cardano",
};
const COIN_COLORS: Record<string, string> = {
  BTC: C.amber,
  ETH: C.blue,
  BNB: C.amber,
  SOL: C.purple,
  ADA: C.cyan,
};

function SparkLine({ data, height = 40 }: { data: number[]; color: string; height?: number }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const w = 120, h = height;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / (max - min || 1)) * h;
      return `${x},${y}`;
    })
    .join(" ");
  const last = data[data.length - 1];
  const prev = data[data.length - 2];
  const up = last >= prev;
  return (
    <svg
      width={w}
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      style={{ overflow: "visible" }}
    >
      <polyline
        points={pts}
        fill="none"
        stroke={up ? C.green : C.red}
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
    </svg>
  );
}

function CandleChart({ data, width = 600, height = 180 }: { data: PricePoint[]; width?: number; height?: number }) {
  if (!data || data.length < 2) return null;
  const n = Math.min(data.length, 60);
  const slice = data.slice(-n);
  const allPrices = slice.flatMap((d) => [d.high, d.low]);
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const range = maxP - minP || 1;
  const padL = 4, padR = 4, padT = 10, padB = 10;
  const chartW = width - padL - padR;
  const chartH = height - padT - padB;
  const cw = chartW / n;
  const scaleY = (v: number) => padT + chartH - ((v - minP) / range) * chartH;
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{ display: "block" }}
    >
      {slice.map((d, i) => {
        const x = padL + i * cw + cw * 0.1;
        const bw = cw * 0.8;
        const up = d.close >= d.open;
        const color = up ? C.green : C.red;
        const top = scaleY(Math.max(d.open, d.close));
        const bot = scaleY(Math.min(d.open, d.close));
        const bh = Math.max(1, bot - top);
        const wickX = x + bw / 2;
        return (
          <g key={i}>
            <line
              x1={wickX}
              y1={scaleY(d.high)}
              x2={wickX}
              y2={scaleY(d.low)}
              stroke={color}
              strokeWidth={0.8}
              opacity={0.6}
            />
            <rect
              x={x}
              y={top}
              width={bw}
              height={bh}
              fill={up ? C.greenDim : C.redDim}
              stroke={color}
              strokeWidth={0.8}
              rx={0.5}
            />
          </g>
        );
      })}
      {[0.25, 0.5, 0.75].map((t) => {
        const yv = minP + range * t;
        const y = scaleY(yv);
        return (
          <g key={t}>
            <line
              x1={padL}
              y1={y}
              x2={width - padR}
              y2={y}
              stroke={C.border}
              strokeWidth={0.5}
              strokeDasharray="3,4"
            />
            <text
              x={width - padR - 2}
              y={y - 3}
              fill={C.muted}
              fontSize={8}
              textAnchor="end"
            >
              ${yv.toFixed(0)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function VolChart({ data, width = 600, height = 80 }: { data: PricePoint[]; width?: number; height?: number }) {
  if (!data || data.length < 2) return null;
  const valid = data.filter((d) => d.vol !== null && d.vol !== undefined);
  if (!valid.length) return null;
  const n = Math.min(valid.length, 60);
  const slice = valid.slice(-n);
  const maxV = Math.max(...slice.map((d) => d.vol as number));
  const chartW = width, chartH = height;
  const bw = chartW / n;
  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{ display: "block" }}
    >
      <line
        x1={0}
        y1={chartH * 0.4}
        x2={chartW}
        y2={chartH * 0.4}
        stroke={C.amber}
        strokeWidth={0.5}
        strokeDasharray="3,4"
        opacity={0.4}
      />
      {slice.map((d, i) => {
        const x = i * bw;
        const h = ((d.vol as number) / maxV) * chartH;
        const color =
          (d.vol as number) > 50 ? C.red : (d.vol as number) > 30 ? C.amber : C.green;
        return (
          <rect
            key={i}
            x={x + 1}
            y={chartH - h}
            width={bw - 2}
            height={h}
            fill={color}
            opacity={0.7}
            rx={1}
          />
        );
      })}
    </svg>
  );
}

function GaugeArc({ value, max = 100, color, size = 90 }: { value: number; max?: number; color: string; size?: number }) {
  const r = 36, cx = 45, cy = 45;
  const startAngle = -210, totalAngle = 240;
  const angle = startAngle + (value / max) * totalAngle;
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const arcPath = (start: number, end: number, radius: number) => {
    const s = {
      x: cx + radius * Math.cos(toRad(start)),
      y: cy + radius * Math.sin(toRad(start)),
    };
    const e = {
      x: cx + radius * Math.cos(toRad(end)),
      y: cy + radius * Math.sin(toRad(end)),
    };
    const large = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${large} 1 ${e.x} ${e.y}`;
  };
  return (
    <svg width={size} height={size} viewBox="0 0 90 90">
      <path
        d={arcPath(-210, 30, r)}
        fill="none"
        stroke={C.border}
        strokeWidth={6}
        strokeLinecap="round"
      />
      <path
        d={arcPath(-210, angle, r)}
        fill="none"
        stroke={color}
        strokeWidth={6}
        strokeLinecap="round"
      />
      <text
        x={cx}
        y={cy + 5}
        textAnchor="middle"
        fill={C.text}
        fontSize={14}
        fontWeight={700}
        fontFamily="monospace"
      >
        {Math.round(value)}
      </text>
      <text
        x={cx}
        y={cy + 16}
        textAnchor="middle"
        fill={C.muted}
        fontSize={7}
      >
        / {max}
      </text>
    </svg>
  );
}

function VolBadge({ label }: { label: string }) {
  const cfg: Record<string, { bg: string; color: string; icon: string }> = {
    Low: { bg: C.greenDim, color: C.green, icon: "▼" },
    Medium: { bg: C.amberDim, color: C.amber, icon: "◆" },
    High: { bg: C.redDim, color: C.red, icon: "▲" },
  };
  const s = cfg[label] ?? cfg.Medium;
  return (
    <span
      style={{
        background: s.bg,
        color: s.color,
        border: `1px solid ${s.color}33`,
        borderRadius: 6,
        padding: "3px 10px",
        fontSize: 12,
        fontWeight: 700,
        letterSpacing: ".04em",
      }}
    >
      {s.icon} {label} Vol
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState("overview");
  const [coin, setCoin] = useState("BTC");
  const [allData, setAllData] = useState<Record<string, PricePoint[]>>({});
  const [model, setModel] = useState<MLModel | null>(null);
  const [modelMeta, setModelMeta] = useState<{ accuracy: number; trainedOn: number; coins: number } | null>(null);
  const [predInputs, setPredInputs] = useState({
    vol: "35",
    rsi: "52",
    priceMa7: "0.01",
    priceMa20: "-0.02",
    ret: "0.005",
  });
  const [predResult, setPredResult] = useState<{ label: string; probs: number[]; confidence: number } | null>(null);
  const [training, setTraining] = useState(false);
  const [trainPct, setTrainPct] = useState(0);
  const [uploadedData, setUploadedData] = useState<{ headers: string[]; rows: Record<string, string>[]; name: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const d: Record<string, PricePoint[]> = {};
    COINS.forEach((c) => {
      const prices = generatePriceHistory(c);
      d[c] = calcFeatures(prices);
    });
    setAllData(d);
  }, []);

  const currentData = allData[coin] ?? [];
  const latest = currentData[currentData.length - 1] ?? ({} as PricePoint);
  const prev = currentData[currentData.length - 2] ?? ({} as PricePoint);
  const priceChange =
    latest.price && prev.price
      ? ((latest.price - prev.price) / prev.price) * 100
      : 0;
  const volChange =
    latest.vol && prev.vol ? (latest.vol as number) - (prev.vol as number) : 0;

  const handleTrain = () => {
    setTraining(true);
    setTrainPct(0);
    const steps = [20, 50, 75, 90, 100];
    let i = 0;
    const iv = setInterval(() => {
      setTrainPct(steps[i]);
      i++;
      if (i >= steps.length) {
        clearInterval(iv);
        const allRows = COINS.flatMap((c) => allData[c] ?? []);
        const trained = trainVolatilityModel(allRows);
        setModel(trained);
        setModelMeta({
          accuracy: trained.accuracy,
          trainedOn: allRows.filter((d) => d.vol !== null).length,
          coins: COINS.length,
        });
        setTraining(false);
      }
    }, 500);
  };

  const handlePredict = () => {
    if (!model) return;
    const f = [
      parseFloat(predInputs.vol) / 100,
      parseFloat(predInputs.rsi) / 100,
      parseFloat(predInputs.priceMa7),
      parseFloat(predInputs.priceMa20),
      parseFloat(predInputs.ret),
    ];
    setPredResult(model.predict(f));
  };

  const handleCSV = (file: File | null | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
      const rows = lines.slice(1).map((l) => {
        const v = l.split(",").map((x) => x.trim().replace(/^"|"$/g, ""));
        const obj: Record<string, string> = {};
        headers.forEach((h, i) => (obj[h] = v[i] ?? ""));
        return obj;
      });
      setUploadedData({ headers, rows, name: file.name });
    };
    reader.readAsText(file);
  };

  const navTabs = [
    { id: "overview", label: "Overview", icon: "◈" },
    { id: "analysis", label: "Volatility Analysis", icon: "⟨/⟩" },
    { id: "model", label: "ML Model", icon: "◎" },
    { id: "predict", label: "Predict", icon: "◆" },
    { id: "data", label: "Data", icon: "⬆" },
  ];

  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        color: C.text,
        fontFamily: "'IBM Plex Mono', 'Fira Mono', monospace",
        fontSize: 13,
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
        rel="stylesheet"
      />

      {/* Nav */}
      <div
        style={{
          borderBottom: `1px solid ${C.border}`,
          background: C.panel,
          padding: "0 28px",
          display: "flex",
          alignItems: "center",
          gap: 0,
          height: 52,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginRight: 32,
          }}
        >
          <span style={{ fontSize: 18, color: C.amber }}>⬡</span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              color: C.text,
              letterSpacing: ".08em",
            }}
          >
            CRYPTO<span style={{ color: C.cyan }}>VOL</span>
          </span>
          <span
            style={{
              fontSize: 10,
              color: C.muted,
              background: C.dim,
              padding: "1px 6px",
              borderRadius: 4,
            }}
          >
            ML
          </span>
        </div>
        <div
          style={{ display: "flex", alignItems: "center", gap: 2, flex: 1 }}
        >
          {navTabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                background: tab === t.id ? C.dim : "transparent",
                color: tab === t.id ? C.cyan : C.muted,
                border: "none",
                borderBottom:
                  tab === t.id
                    ? `2px solid ${C.cyan}`
                    : "2px solid transparent",
                padding: "0 16px",
                height: 52,
                cursor: "pointer",
                fontSize: 12,
                fontFamily: "inherit",
                fontWeight: tab === t.id ? 600 : 400,
                letterSpacing: ".04em",
                transition: "all .15s",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          {COINS.map((c) => (
            <button
              key={c}
              onClick={() => setCoin(c)}
              style={{
                background: coin === c ? `${COIN_COLORS[c]}22` : "transparent",
                color: coin === c ? COIN_COLORS[c] : C.muted,
                border: `1px solid ${coin === c ? COIN_COLORS[c] + "55" : C.border}`,
                borderRadius: 6,
                padding: "4px 10px",
                cursor: "pointer",
                fontSize: 11,
                fontFamily: "inherit",
                fontWeight: coin === c ? 600 : 400,
                transition: "all .15s",
              }}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "24px 28px" }}>

        {/* OVERVIEW TAB */}
        {tab === "overview" && (
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: 12,
                marginBottom: 24,
              }}
            >
              <span
                style={{
                  fontSize: 22,
                  fontWeight: 600,
                  color: COIN_COLORS[coin],
                }}
              >
                {COIN_NAMES[coin]}
              </span>
              <span style={{ fontSize: 13, color: C.muted }}>
                {coin} · Simulated 90-day data
              </span>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 14,
                marginBottom: 24,
              }}
            >
              {[
                {
                  label: "Price",
                  value: `$${latest.price?.toFixed(2) ?? "—"}`,
                  sub: `${priceChange >= 0 ? "+" : ""}${priceChange.toFixed(2)}%`,
                  color: priceChange >= 0 ? C.green : C.red,
                },
                {
                  label: "Volatility (20d ann.)",
                  value: latest.vol ? `${(latest.vol as number).toFixed(1)}%` : "—",
                  sub: `${volChange >= 0 ? "+" : ""}${volChange.toFixed(1)}% vs prev`,
                  color:
                    (latest.vol as number) > 50
                      ? C.red
                      : (latest.vol as number) > 30
                      ? C.amber
                      : C.green,
                },
                {
                  label: "RSI (14)",
                  value: latest.rsi ? latest.rsi.toFixed(1) : "—",
                  sub:
                    (latest.rsi ?? 0) > 70
                      ? "Overbought"
                      : (latest.rsi ?? 0) < 30
                      ? "Oversold"
                      : "Neutral",
                  color:
                    (latest.rsi ?? 0) > 70
                      ? C.red
                      : (latest.rsi ?? 0) < 30
                      ? C.green
                      : C.amber,
                },
                {
                  label: "Vol regime",
                  value: latest.vol
                    ? (latest.vol as number) > 50
                      ? "HIGH"
                      : (latest.vol as number) > 30
                      ? "MED"
                      : "LOW"
                    : "—",
                  sub: "20-day window",
                  color:
                    (latest.vol as number) > 50
                      ? C.red
                      : (latest.vol as number) > 30
                      ? C.amber
                      : C.green,
                },
              ].map((card) => (
                <div
                  key={card.label}
                  style={{
                    background: C.panel,
                    border: `1px solid ${C.border}`,
                    borderRadius: 12,
                    padding: "16px 18px",
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      color: C.muted,
                      letterSpacing: ".08em",
                      marginBottom: 8,
                      textTransform: "uppercase",
                    }}
                  >
                    {card.label}
                  </div>
                  <div
                    style={{
                      fontSize: 24,
                      fontWeight: 600,
                      color: card.color,
                      lineHeight: 1,
                    }}
                  >
                    {card.value}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: card.color,
                      opacity: 0.7,
                      marginTop: 6,
                    }}
                  >
                    {card.sub}
                  </div>
                </div>
              ))}
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "2fr 1fr",
                gap: 14,
                marginBottom: 14,
              }}
            >
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "18px 18px 12px",
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: C.muted,
                    letterSpacing: ".08em",
                    marginBottom: 12,
                    textTransform: "uppercase",
                  }}
                >
                  Price · 60-day candles
                </div>
                <CandleChart data={currentData} width={560} height={180} />
              </div>
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "18px",
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: C.muted,
                    letterSpacing: ".08em",
                    marginBottom: 16,
                    textTransform: "uppercase",
                  }}
                >
                  Market indicators
                </div>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 14,
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ color: C.muted, fontSize: 11 }}>
                      MA7 / MA20 spread
                    </span>
                    <span
                      style={{
                        color:
                          (latest.priceMa7Ratio ?? 0) > 0 ? C.green : C.red,
                        fontSize: 12,
                        fontWeight: 600,
                      }}
                    >
                      {latest.priceMa7Ratio
                        ? `${(latest.priceMa7Ratio * 100).toFixed(2)}%`
                        : "—"}
                    </span>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ color: C.muted, fontSize: 11 }}>
                      Daily return
                    </span>
                    <span
                      style={{
                        color: latest.ret > 0 ? C.green : C.red,
                        fontSize: 12,
                        fontWeight: 600,
                      }}
                    >
                      {latest.ret
                        ? `${(latest.ret * 100).toFixed(3)}%`
                        : "—"}
                    </span>
                  </div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span style={{ color: C.muted, fontSize: 11 }}>
                      Vol rank
                    </span>
                    <span
                      style={{
                        color: C.cyan,
                        fontSize: 12,
                        fontWeight: 600,
                      }}
                    >
                      {latest.volRank
                        ? `${(latest.volRank * 100).toFixed(0)}th pct`
                        : "—"}
                    </span>
                  </div>
                  <div
                    style={{
                      borderTop: `1px solid ${C.border}`,
                      paddingTop: 14,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        marginBottom: 8,
                      }}
                    >
                      COMPARE ALL COINS · 7d change
                    </div>
                    {COINS.map((c) => {
                      const d = allData[c] ?? [];
                      const last = d[d.length - 1]?.price;
                      const seven = d[d.length - 8]?.price;
                      const chg =
                        last && seven ? ((last - seven) / seven) * 100 : 0;
                      return (
                        <div
                          key={c}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            marginBottom: 6,
                          }}
                        >
                          <span
                            style={{
                              color: COIN_COLORS[c],
                              fontSize: 11,
                              width: 34,
                            }}
                          >
                            {c}
                          </span>
                          <div
                            style={{
                              flex: 1,
                              height: 4,
                              background: C.dim,
                              borderRadius: 2,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.min(100, Math.abs(chg) * 5)}%`,
                                height: "100%",
                                background: chg >= 0 ? C.green : C.red,
                                borderRadius: 2,
                              }}
                            />
                          </div>
                          <span
                            style={{
                              fontSize: 11,
                              color: chg >= 0 ? C.green : C.red,
                              minWidth: 48,
                              textAlign: "right",
                            }}
                          >
                            {chg >= 0 ? "+" : ""}
                            {chg.toFixed(2)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ANALYSIS TAB */}
        {tab === "analysis" && (
          <div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                marginBottom: 20,
                color: C.cyan,
              }}
            >
              Volatility analysis · {coin}
            </div>
            <div
              style={{
                background: C.panel,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: "18px 18px 10px",
                marginBottom: 14,
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: C.muted,
                  letterSpacing: ".08em",
                  marginBottom: 10,
                  textTransform: "uppercase",
                }}
              >
                Realized volatility (20-day annualized) · 60 sessions
              </div>
              <VolChart data={currentData} width={800} height={100} />
              <div
                style={{
                  display: "flex",
                  gap: 16,
                  marginTop: 10,
                  fontSize: 10,
                  color: C.muted,
                }}
              >
                <span style={{ color: C.green }}>■ Low (&lt;30%)</span>
                <span style={{ color: C.amber }}>■ Medium (30–50%)</span>
                <span style={{ color: C.red }}>■ High (&gt;50%)</span>
              </div>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 14,
                marginBottom: 14,
              }}
            >
              {COINS.map((c) => {
                const d = allData[c] ?? [];
                const last = d[d.length - 1] ?? ({} as PricePoint);
                const vols = d
                  .filter((x) => x.vol !== null && x.vol !== undefined)
                  .map((x) => x.vol as number);
                const avgVol = vols.length
                  ? vols.reduce((a, b) => a + b, 0) / vols.length
                  : 0;
                const maxVol = vols.length ? Math.max(...vols) : 0;
                const regime =
                  (last.vol as number) > 50
                    ? "High"
                    : (last.vol as number) > 30
                    ? "Medium"
                    : "Low";
                return (
                  <div
                    key={c}
                    style={{
                      background: C.panel,
                      border: `1px solid ${c === coin ? COIN_COLORS[c] + "55" : C.border}`,
                      borderRadius: 12,
                      padding: "16px 18px",
                      cursor: "pointer",
                    }}
                    onClick={() => setCoin(c)}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: 12,
                      }}
                    >
                      <span
                        style={{
                          color: COIN_COLORS[c],
                          fontWeight: 600,
                          fontSize: 14,
                        }}
                      >
                        {c}
                      </span>
                      <VolBadge label={regime} />
                    </div>
                    <SparkLine
                      data={vols.slice(-30)}
                      color={COIN_COLORS[c]}
                      height={36}
                    />
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        marginTop: 10,
                        fontSize: 11,
                        color: C.muted,
                      }}
                    >
                      <span>
                        avg{" "}
                        <span style={{ color: C.text }}>
                          {avgVol.toFixed(1)}%
                        </span>
                      </span>
                      <span>
                        cur{" "}
                        <span style={{ color: C.text }}>
                          {(last.vol as number | undefined)?.toFixed(1) ?? "—"}%
                        </span>
                      </span>
                      <span>
                        max{" "}
                        <span style={{ color: C.text }}>
                          {maxVol.toFixed(1)}%
                        </span>
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div
              style={{
                background: C.panel,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: "18px",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: C.muted,
                  letterSpacing: ".08em",
                  marginBottom: 14,
                  textTransform: "uppercase",
                }}
              >
                RSI · {coin} (14-day)
              </div>
              <div style={{ position: "relative" }}>
                <svg
                  width="100%"
                  viewBox="0 0 800 70"
                  preserveAspectRatio="none"
                  style={{ display: "block" }}
                >
                  {[30, 50, 70].map((lvl) => (
                    <g key={lvl}>
                      <line
                        x1={0}
                        y1={70 - (lvl / 100) * 70}
                        x2={800}
                        y2={70 - (lvl / 100) * 70}
                        stroke={
                          lvl === 50
                            ? C.dim
                            : lvl === 70
                            ? C.redDim
                            : C.greenDim
                        }
                        strokeWidth={1}
                        strokeDasharray="4,4"
                      />
                      <text
                        x={4}
                        y={70 - (lvl / 100) * 70 - 2}
                        fill={C.muted}
                        fontSize={8}
                      >
                        {lvl}
                      </text>
                    </g>
                  ))}
                  {(() => {
                    const d = currentData.filter(
                      (x) => x.rsi !== null && x.rsi !== undefined
                    );
                    const n = Math.min(d.length, 60);
                    const sl = d.slice(-n);
                    const pts = sl
                      .map(
                        (p, i) =>
                          `${(i / (n - 1)) * 800},${70 - ((p.rsi ?? 50) / 100) * 70}`
                      )
                      .join(" ");
                    return (
                      <polyline
                        points={pts}
                        fill="none"
                        stroke={C.purple}
                        strokeWidth={1.5}
                        strokeLinejoin="round"
                      />
                    );
                  })()}
                </svg>
              </div>
            </div>
          </div>
        )}

        {/* ML MODEL TAB */}
        {tab === "model" && (
          <div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                marginBottom: 6,
                color: C.cyan,
              }}
            >
              ML model · Volatility regime classifier
            </div>
            <div
              style={{
                fontSize: 12,
                color: C.muted,
                marginBottom: 24,
              }}
            >
              Logistic Regression · Softmax · 3 classes (Low / Medium / High
              volatility)
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
                marginBottom: 20,
              }}
            >
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "20px",
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: C.muted,
                    textTransform: "uppercase",
                    letterSpacing: ".08em",
                    marginBottom: 14,
                  }}
                >
                  Feature engineering
                </div>
                {[
                  {
                    name: "vol_rank",
                    desc: "20-day annualized volatility (normalized)",
                    color: C.amber,
                  },
                  {
                    name: "rsi_14",
                    desc: "Relative strength index / 100",
                    color: C.purple,
                  },
                  {
                    name: "price_ma7_ratio",
                    desc: "Price deviation from 7-day MA",
                    color: C.blue,
                  },
                  {
                    name: "price_ma20_ratio",
                    desc: "Price deviation from 20-day MA",
                    color: C.cyan,
                  },
                  {
                    name: "daily_return",
                    desc: "Log return for the session",
                    color: C.green,
                  },
                ].map((f) => (
                  <div
                    key={f.name}
                    style={{
                      display: "flex",
                      gap: 10,
                      alignItems: "flex-start",
                      marginBottom: 12,
                    }}
                  >
                    <span
                      style={{
                        color: f.color,
                        fontSize: 11,
                        fontWeight: 600,
                        minWidth: 130,
                      }}
                    >
                      {f.name}
                    </span>
                    <span style={{ color: C.muted, fontSize: 11 }}>
                      {f.desc}
                    </span>
                  </div>
                ))}
              </div>
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "20px",
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: C.muted,
                    textTransform: "uppercase",
                    letterSpacing: ".08em",
                    marginBottom: 14,
                  }}
                >
                  Pipeline
                </div>
                {[
                  {
                    step: "01",
                    label: "Data generation",
                    detail: "90-day OHLCV per coin",
                  },
                  {
                    step: "02",
                    label: "Feature calc",
                    detail: "MA, RSI, rolling vol",
                  },
                  {
                    step: "03",
                    label: "Normalization",
                    detail: "Min-max per feature",
                  },
                  {
                    step: "04",
                    label: "Label encoding",
                    detail: "Low / Med / High vol",
                  },
                  {
                    step: "05",
                    label: "Logistic Regression",
                    detail: "Softmax · 400 epochs",
                  },
                  {
                    step: "06",
                    label: "Evaluation",
                    detail: "Accuracy on train set",
                  },
                ].map((s) => (
                  <div
                    key={s.step}
                    style={{
                      display: "flex",
                      gap: 12,
                      alignItems: "center",
                      marginBottom: 10,
                    }}
                  >
                    <span
                      style={{
                        color: C.muted,
                        fontSize: 10,
                        minWidth: 20,
                      }}
                    >
                      {s.step}
                    </span>
                    <span style={{ color: C.border, fontSize: 10 }}>→</span>
                    <span
                      style={{
                        color: C.text,
                        fontSize: 11,
                        fontWeight: 500,
                      }}
                    >
                      {s.label}
                    </span>
                    <span
                      style={{
                        color: C.muted,
                        fontSize: 11,
                        marginLeft: "auto",
                      }}
                    >
                      {s.detail}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {!model ? (
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "28px",
                  textAlign: "center",
                }}
              >
                {training ? (
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        color: C.muted,
                        marginBottom: 14,
                      }}
                    >
                      Training across {COINS.length} coins · {trainPct}%
                    </div>
                    <div
                      style={{
                        height: 6,
                        background: C.dim,
                        borderRadius: 3,
                        overflow: "hidden",
                        maxWidth: 400,
                        margin: "0 auto",
                      }}
                    >
                      <div
                        style={{
                          width: `${trainPct}%`,
                          height: "100%",
                          background: C.cyan,
                          borderRadius: 3,
                          transition: "width .4s ease",
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        color: C.muted,
                        marginBottom: 18,
                      }}
                    >
                      Model not trained yet. Click below to train on all{" "}
                      {COINS.length} coins.
                    </div>
                    <button
                      onClick={handleTrain}
                      style={{
                        background: C.cyan + "22",
                        color: C.cyan,
                        border: `1px solid ${C.cyan}55`,
                        borderRadius: 8,
                        padding: "12px 32px",
                        fontSize: 13,
                        cursor: "pointer",
                        fontFamily: "inherit",
                        fontWeight: 600,
                      }}
                    >
                      ⬡ Train model
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.green}33`,
                  borderRadius: 12,
                  padding: "24px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    marginBottom: 20,
                  }}
                >
                  <span style={{ color: C.green, fontSize: 16 }}>✓</span>
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: C.green,
                    }}
                  >
                    Model trained
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 14,
                  }}
                >
                  <div
                    style={{
                      background: C.bg,
                      borderRadius: 10,
                      padding: "14px 16px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        marginBottom: 6,
                      }}
                    >
                      ACCURACY
                    </div>
                    <div
                      style={{
                        fontSize: 28,
                        fontWeight: 700,
                        color: C.green,
                      }}
                    >
                      {((modelMeta?.accuracy ?? 0) * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div
                    style={{
                      background: C.bg,
                      borderRadius: 10,
                      padding: "14px 16px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        marginBottom: 6,
                      }}
                    >
                      TRAINING SAMPLES
                    </div>
                    <div
                      style={{
                        fontSize: 28,
                        fontWeight: 700,
                        color: C.cyan,
                      }}
                    >
                      {modelMeta?.trainedOn.toLocaleString()}
                    </div>
                  </div>
                  <div
                    style={{
                      background: C.bg,
                      borderRadius: 10,
                      padding: "14px 16px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        marginBottom: 6,
                      }}
                    >
                      COINS
                    </div>
                    <div
                      style={{
                        fontSize: 28,
                        fontWeight: 700,
                        color: C.amber,
                      }}
                    >
                      {modelMeta?.coins}
                    </div>
                  </div>
                </div>
                <div
                  style={{
                    marginTop: 16,
                    fontSize: 12,
                    color: C.muted,
                  }}
                >
                  Classes:{" "}
                  <span style={{ color: C.green }}>Low</span> ·{" "}
                  <span style={{ color: C.amber }}>Medium</span> ·{" "}
                  <span style={{ color: C.red }}>High</span> volatility regime
                  · Task: multi-class classification
                </div>
              </div>
            )}
          </div>
        )}

        {/* PREDICT TAB */}
        {tab === "predict" && (
          <div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                marginBottom: 6,
                color: C.cyan,
              }}
            >
              Volatility regime predictor
            </div>
            <div
              style={{
                fontSize: 12,
                color: C.muted,
                marginBottom: 24,
              }}
            >
              Enter current market features → get predicted volatility regime
            </div>

            {!model && (
              <div
                style={{
                  background: C.amberDim,
                  border: `1px solid ${C.amber}44`,
                  borderRadius: 10,
                  padding: "12px 16px",
                  marginBottom: 20,
                  fontSize: 12,
                  color: C.amber,
                }}
              >
                ⚠ Train the model first on the ML Model tab before predicting.
              </div>
            )}

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 14,
                marginBottom: 20,
              }}
            >
              {[
                {
                  key: "vol" as const,
                  label: "Realized volatility (%)",
                  hint: "Annualized 20-day vol, e.g. 35",
                  min: 0,
                  max: 120,
                  color: C.amber,
                },
                {
                  key: "rsi" as const,
                  label: "RSI (14-day)",
                  hint: "0–100 scale",
                  min: 0,
                  max: 100,
                  color: C.purple,
                },
                {
                  key: "priceMa7" as const,
                  label: "Price vs MA7 ratio",
                  hint: "e.g. 0.012 means +1.2% above MA7",
                  min: -0.2,
                  max: 0.2,
                  color: C.blue,
                },
                {
                  key: "priceMa20" as const,
                  label: "Price vs MA20 ratio",
                  hint: "e.g. -0.03 means 3% below MA20",
                  min: -0.3,
                  max: 0.3,
                  color: C.cyan,
                },
                {
                  key: "ret" as const,
                  label: "Daily return",
                  hint: "e.g. 0.005 = +0.5%",
                  min: -0.15,
                  max: 0.15,
                  color: C.green,
                },
              ].map((f) => (
                <div
                  key={f.key}
                  style={{
                    background: C.panel,
                    border: `1px solid ${C.border}`,
                    borderRadius: 12,
                    padding: "16px 18px",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: 8,
                    }}
                  >
                    <span
                      style={{
                        color: f.color,
                        fontWeight: 600,
                        fontSize: 12,
                      }}
                    >
                      {f.label}
                    </span>
                    <span style={{ color: C.muted, fontSize: 11 }}>
                      {f.hint}
                    </span>
                  </div>
                  <input
                    type="number"
                    value={predInputs[f.key]}
                    onChange={(e) =>
                      setPredInputs((p) => ({ ...p, [f.key]: e.target.value }))
                    }
                    style={{
                      width: "100%",
                      background: C.bg,
                      color: C.text,
                      border: `1px solid ${C.border}`,
                      borderRadius: 8,
                      padding: "9px 12px",
                      fontSize: 14,
                      fontFamily: "inherit",
                      marginBottom: 8,
                      boxSizing: "border-box",
                    }}
                  />
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={(f.max - f.min) / 200}
                    value={predInputs[f.key] || 0}
                    onChange={(e) =>
                      setPredInputs((p) => ({
                        ...p,
                        [f.key]: parseFloat(e.target.value).toFixed(4),
                      }))
                    }
                    style={{ width: "100%", accentColor: f.color }}
                  />
                </div>
              ))}
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.border}`,
                  borderRadius: 12,
                  padding: "16px 18px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                  alignItems: "center",
                  gap: 10,
                }}
              >
                <div style={{ fontSize: 11, color: C.muted }}>
                  Quick fill from live data
                </div>
                {COINS.map((c) => {
                  const d = allData[c] ?? [];
                  const last = d[d.length - 1] ?? ({} as PricePoint);
                  return (
                    <button
                      key={c}
                      onClick={() => {
                        setPredInputs({
                          vol: last.vol?.toFixed(2) ?? "35",
                          rsi: last.rsi?.toFixed(2) ?? "52",
                          priceMa7: last.priceMa7Ratio?.toFixed(4) ?? "0",
                          priceMa20: last.priceMa20Ratio?.toFixed(4) ?? "0",
                          ret: last.ret?.toFixed(5) ?? "0",
                        });
                        setCoin(c);
                      }}
                      style={{
                        background: `${COIN_COLORS[c]}15`,
                        color: COIN_COLORS[c],
                        border: `1px solid ${COIN_COLORS[c]}44`,
                        borderRadius: 6,
                        padding: "6px 16px",
                        cursor: "pointer",
                        fontSize: 12,
                        fontFamily: "inherit",
                        width: "100%",
                        fontWeight: 500,
                      }}
                    >
                      ⬇ Fill from {c}
                    </button>
                  );
                })}
              </div>
            </div>

            <button
              onClick={handlePredict}
              disabled={!model}
              style={{
                background: model ? `${C.cyan}22` : C.dim,
                color: model ? C.cyan : C.muted,
                border: `1px solid ${model ? C.cyan + "55" : C.border}`,
                borderRadius: 10,
                padding: "14px",
                fontSize: 14,
                cursor: model ? "pointer" : "not-allowed",
                fontFamily: "inherit",
                fontWeight: 700,
                width: "100%",
                marginBottom: 20,
                letterSpacing: ".06em",
              }}
            >
              ◆ PREDICT VOLATILITY REGIME
            </button>

            {predResult && (
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${
                    predResult.label === "High"
                      ? C.red
                      : predResult.label === "Medium"
                      ? C.amber
                      : C.green
                  }44`,
                  borderRadius: 14,
                  padding: "28px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    gap: 28,
                    alignItems: "flex-start",
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        textTransform: "uppercase",
                        letterSpacing: ".1em",
                        marginBottom: 10,
                      }}
                    >
                      Predicted regime
                    </div>
                    <VolBadge label={predResult.label} />
                    <div style={{ marginTop: 12 }}>
                      <GaugeArc
                        value={Math.round(predResult.confidence * 100)}
                        max={100}
                        color={
                          predResult.label === "High"
                            ? C.red
                            : predResult.label === "Medium"
                            ? C.amber
                            : C.green
                        }
                        size={90}
                      />
                    </div>
                  </div>
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div
                      style={{
                        fontSize: 10,
                        color: C.muted,
                        textTransform: "uppercase",
                        letterSpacing: ".08em",
                        marginBottom: 14,
                      }}
                    >
                      Class probabilities
                    </div>
                    {["Low", "Medium", "High"].map((cls, i) => {
                      const prob = predResult.probs[i] ?? 0;
                      const col = [C.green, C.amber, C.red][i];
                      return (
                        <div key={cls} style={{ marginBottom: 12 }}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              marginBottom: 5,
                              fontSize: 11,
                            }}
                          >
                            <span style={{ color: col, fontWeight: 600 }}>
                              {cls}
                            </span>
                            <span style={{ color: C.text }}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div
                            style={{
                              height: 6,
                              background: C.dim,
                              borderRadius: 3,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.round(prob * 100)}%`,
                                height: "100%",
                                background: col,
                                borderRadius: 3,
                                transition: "width .5s ease",
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                    <div
                      style={{
                        marginTop: 18,
                        padding: "12px 14px",
                        background: C.bg,
                        borderRadius: 10,
                        fontSize: 11,
                        color: C.muted,
                      }}
                    >
                      {predResult.label === "High" &&
                        "⚠ High volatility regime detected. Expect large price swings. Consider tighter risk controls."}
                      {predResult.label === "Medium" &&
                        "◆ Moderate volatility. Normal trading conditions. Standard position sizing recommended."}
                      {predResult.label === "Low" &&
                        "✓ Low volatility environment. Market is calm. May be suitable for trend-following strategies."}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* DATA TAB */}
        {tab === "data" && (
          <div>
            <div
              style={{
                fontSize: 16,
                fontWeight: 600,
                marginBottom: 6,
                color: C.cyan,
              }}
            >
              Data
            </div>
            <div
              style={{
                fontSize: 12,
                color: C.muted,
                marginBottom: 24,
              }}
            >
              Upload your own crypto OHLCV CSV or browse the generated dataset.
            </div>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                handleCSV(e.dataTransfer.files[0]);
              }}
              onClick={() => fileRef.current?.click()}
              style={{
                border: `1.5px dashed ${C.borderHi}`,
                borderRadius: 14,
                padding: "36px",
                textAlign: "center",
                cursor: "pointer",
                background: C.panel,
                marginBottom: 20,
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 10 }}>⬆</div>
              <div
                style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}
              >
                Drop your CSV here
              </div>
              <div style={{ fontSize: 12, color: C.muted }}>
                Expected columns: date, open, high, low, close, volume
              </div>
              <input
                ref={fileRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={(e) => handleCSV(e.target.files?.[0])}
              />
            </div>

            {uploadedData && (
              <div
                style={{
                  background: C.panel,
                  border: `1px solid ${C.green}44`,
                  borderRadius: 12,
                  padding: "20px",
                  marginBottom: 20,
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    color: C.green,
                    marginBottom: 12,
                  }}
                >
                  ✓ {uploadedData.name} · {uploadedData.rows.length} rows ·{" "}
                  {uploadedData.headers.length} columns
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: 11,
                    }}
                  >
                    <thead>
                      <tr>
                        {uploadedData.headers.map((h) => (
                          <th
                            key={h}
                            style={{
                              textAlign: "left",
                              padding: "6px 10px",
                              color: C.muted,
                              borderBottom: `1px solid ${C.border}`,
                            }}
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {uploadedData.rows.slice(0, 5).map((r, i) => (
                        <tr key={i}>
                          {uploadedData.headers.map((h) => (
                            <td
                              key={h}
                              style={{
                                padding: "6px 10px",
                                borderBottom: `1px solid ${C.border}`,
                              }}
                            >
                              {r[h]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <div
              style={{
                background: C.panel,
                border: `1px solid ${C.border}`,
                borderRadius: 12,
                padding: "20px",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: C.muted,
                  textTransform: "uppercase",
                  letterSpacing: ".08em",
                  marginBottom: 14,
                }}
              >
                Generated dataset · {coin} (last 10 rows)
              </div>
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: 11,
                  }}
                >
                  <thead>
                    <tr>
                      {["day", "price", "vol%", "rsi", "ma7", "ma20", "ret%", "regime"].map(
                        (h) => (
                          <th
                            key={h}
                            style={{
                              textAlign: "left",
                              padding: "6px 10px",
                              color: C.muted,
                              borderBottom: `1px solid ${C.border}`,
                            }}
                          >
                            {h}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {currentData.slice(-10).map((r, i) => {
                      const regime =
                        (r.vol as number) > 50
                          ? "High"
                          : (r.vol as number) > 30
                          ? "Medium"
                          : "Low";
                      const rc: Record<string, string> = {
                        High: C.red,
                        Medium: C.amber,
                        Low: C.green,
                      };
                      return (
                        <tr key={i}>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color: C.muted,
                            }}
                          >
                            {r.day}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                            }}
                          >
                            ${r.price?.toFixed(2)}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color:
                                (r.vol as number) > 50
                                  ? C.red
                                  : (r.vol as number) > 30
                                  ? C.amber
                                  : C.green,
                            }}
                          >
                            {(r.vol as number | undefined)?.toFixed(1) ?? "—"}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color:
                                (r.rsi ?? 50) > 70
                                  ? C.red
                                  : (r.rsi ?? 50) < 30
                                  ? C.green
                                  : C.text,
                            }}
                          >
                            {r.rsi?.toFixed(1)}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color: C.muted,
                            }}
                          >
                            ${r.ma7?.toFixed(2)}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color: C.muted,
                            }}
                          >
                            ${r.ma20?.toFixed(2)}
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color: r.ret > 0 ? C.green : C.red,
                            }}
                          >
                            {((r.ret ?? 0) * 100).toFixed(3)}%
                          </td>
                          <td
                            style={{
                              padding: "6px 10px",
                              borderBottom: `1px solid ${C.border}`,
                              color: rc[regime],
                              fontWeight: 600,
                            }}
                          >
                            {regime}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
