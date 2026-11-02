// Headless-Chrome scraper for Swisslos Sporttip football odds.
// Drives the system Chrome via puppeteer-core: loads the SPA, expands every
// collapsed market panel, then reads (market, selection, odds) from the DOM.
//
// Usage: node scraper/scrape_swiss.mjs "<event-url>" [out.csv]
// Writes the CSV (default data/swiss_scraped.csv) plus a *_raw.json with element ids.
// Set CHROME_PATH if Chrome is not at /usr/bin/google-chrome-stable.
import puppeteer from "puppeteer-core";
import fs from "fs";
import path from "path";

const URL = process.argv[2] ||
  "https://www.swisslos.ch/en/sporttip/sports/football/wc-2026/canada-vs-bosnia-herzegovina?t=17812908&tab=filter_1";
const OUT = process.argv[3] || "data/swiss_scraped.csv";
const RAW = OUT.replace(/\.csv$/, "") + "_raw.json";
const CHROME = process.env.CHROME_PATH || "/usr/bin/google-chrome-stable";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1400,2000"],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 2000 });
  console.error("navigating…");
  await page.goto(URL, { waitUntil: "networkidle2", timeout: 90000 });

  // Best-effort: dismiss a cookie/consent banner if one is blocking clicks.
  await sleep(1500);
  for (const sel of ['button#onetrust-accept-btn-handler', 'button[aria-label*="ccept"]',
                     'button[title*="ccept"]', '.cookie button', '#cmpwelcomebtnyes']) {
    try { const b = await page.$(sel); if (b) { await b.click(); console.error("dismissed banner:", sel); await sleep(800); } } catch {}
  }

  // Wait until the marketboard has rendered.
  await page.waitForSelector('asw-marketboard-market', { timeout: 60000 });

  // Expand every collapsed panel. Lazy-loading means new panels/odds appear
  // after each click round, so loop until no collapsed headers remain.
  let round = 0;
  while (round++ < 12) {
    const clicked = await page.evaluate(async () => {
      const headers = [...document.querySelectorAll('[role="button"][aria-expanded="false"]')];
      for (const h of headers) { h.scrollIntoView({ block: "center" }); h.click(); }
      return headers.length;
    });
    console.error(`round ${round}: expanded ${clicked} panels`);
    await sleep(1200);
    if (clicked === 0) break;
  }
  await sleep(1500); // let final odds settle

  // Extract structured data from the live DOM.
  const data = await page.evaluate(() => {
    const out = [];
    const titleOf = (el) => {
      // Climb to the titled market container; its header is the first
      // role=button span. (Grid markets nest a market-block inside it.)
      const host = el.closest("asw-marketboard-market") ||
                   el.closest("asw-marketboard-market-block");
      const hdr = host && host.querySelector('[role="button"] span');
      return hdr ? hdr.textContent.replace(/\s+/g, " ").trim() : "";
    };
    // Pull the line/parameter out of an id, e.g. "...?total=2.5_odds" -> "2.5",
    // "...?hcp=-0.5_odds" -> "-0.5". This is how Over/Under & handicap encode lines.
    const lineOf = (id) => {
      const q = id.indexOf("?");
      if (q < 0) return "";
      let params = id.slice(q + 1).replace(/_odds$/, "");
      const vals = params.split("&").map((kv) => kv.split("=")[1]).filter(Boolean);
      return vals.join(" ");
    };
    // Both id patterns: normal "..._selection_odds" and grid "...?total=X_odds".
    const odds = [...document.querySelectorAll('[id$="_odds"]')];
    for (const o of odds) {
      const val = o.textContent.replace(/\s+/g, " ").trim();
      if (!/^\d+(\.\d+)?$/.test(val)) continue;            // skip blank/suspended/indicators
      const nameEl = document.getElementById(o.id.replace(/_odds$/, "_name"));
      let name = nameEl ? nameEl.textContent.replace(/\s+/g, " ").trim() : "";
      const line = lineOf(o.id);
      if (line) name = name ? `${name} ${line}` : line;    // fold the line into the label
      out.push({ market: titleOf(o), selection: name, odds: parseFloat(val), oddsId: o.id });
    }
    return out;
  });

  // De-duplicate (a selection can appear once per market); keep first.
  const seen = new Set();
  const rows = [];
  for (const r of data) {
    const k = r.oddsId;
    if (seen.has(k)) continue;
    seen.add(k);
    rows.push(r);
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(RAW, JSON.stringify(rows, null, 2));
  const esc = (s) => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  const csv = ["market,selection,decimal_odds",
    ...rows.map((r) => `${esc(r.market)},${esc(r.selection)},${r.odds.toFixed(2)}`)].join("\n");
  fs.writeFileSync(OUT, csv + "\n");

  const markets = new Set(rows.map((r) => r.market));
  console.error(`\nDONE: ${rows.length} selections across ${markets.size} markets -> ${OUT}`);
} finally {
  await browser.close();
}
