import { spawn } from "child_process";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const ROOT_DIR = dirname(fileURLToPath(import.meta.url));

// P10.6 E1: --skip-index пропускает шаги ingest + build_index (только аналитика).
// P13.3: --skip-bitrix пропускает Phase 1 (raw CSV pull + repair), оставляя
//        текущий снимок RAG_DATA нетронутым. Это позволяет переиндексировать
//        ровно тот корпус, на котором собраны regression-кейсы (eval/regression_*),
//        не подмешивая свежие данные из Bitrix.
// --python PATH задаёт кастомный Python-интерпретатор (default: "python").
const ARGV = process.argv.slice(2);
const SKIP_INDEX = ARGV.includes("--skip-index");
const SKIP_BITRIX = ARGV.includes("--skip-bitrix");
const pythonFlagIdx = ARGV.indexOf("--python");
const PYTHON_CMD = pythonFlagIdx >= 0 ? ARGV[pythonFlagIdx + 1] : "D:/SALES_RAG/RAG_RUNTIME/.venv/Scripts/python.exe";
// --batch-size для build_index.py (по умолчанию 8 для GPU — из memory).
const batchFlagIdx = ARGV.indexOf("--batch-size");
const BATCH_SIZE = batchFlagIdx >= 0 ? ARGV[batchFlagIdx + 1] : "8";

function runStep(label, relativeScriptPath) {
  return new Promise((resolvePromise, rejectPromise) => {
    console.log(`\n=== ${label} ===`);
    const child = spawn(process.execPath, [resolve(ROOT_DIR, relativeScriptPath)], {
      cwd: ROOT_DIR,
      stdio: "inherit",
      env: process.env,
    });

    child.on("error", rejectPromise);
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }

      rejectPromise(new Error(`${label} failed with exit code ${code}`));
    });
  });
}

function runPythonStep(label, relativeScriptPath, scriptArgs = []) {
  return new Promise((resolvePromise, rejectPromise) => {
    console.log(`\n=== ${label} ===`);
    const child = spawn(PYTHON_CMD, [resolve(ROOT_DIR, relativeScriptPath), ...scriptArgs], {
      cwd: ROOT_DIR,
      stdio: "inherit",
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    });

    child.on("error", rejectPromise);
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise();
        return;
      }

      rejectPromise(new Error(`${label} failed with exit code ${code}`));
    });
  });
}

async function main() {
  // Phase 1: raw CSV + analytics (JS)
  if (!SKIP_BITRIX) {
    await runStep("Refreshing raw CSV from Bitrix24", "RAG_DATA/generateRagData.mjs");
    await runStep("Repairing orders direction", "RAG_DATA/repairOrdersDirection.mjs");
  } else {
    console.log("\n[--skip-bitrix] Bitrix CSV pull + orders repair пропущены — используем текущий снимок RAG_DATA");
  }
  await runStep("Rebuilding analytics outputs", "RAG_ANALYTICS/runAll.mjs");

  if (!SKIP_INDEX) {
    // Phase 2: ingest Python scripts → data/*.jsonl
    // ВАЖНО: ingest_roadmaps идёт ПЕРЕД ingest.py, потому что product builder
    // (ingest.py) читает roadmap_docs.jsonl для related_roadmap_slugs (B6).
    await runPythonStep("Ingest roadmaps", "RAG_RUNTIME/scripts/ingest_roadmaps.py");
    await runPythonStep("Ingest core docs (product/bundle/deals/…)", "RAG_RUNTIME/scripts/ingest.py");
    await runPythonStep("Ingest knowledge (FAQ + MD + DOCX)", "RAG_RUNTIME/scripts/ingest_knowledge.py");
    await runPythonStep("Ingest offer compositions", "RAG_RUNTIME/scripts/ingest_offer_compositions.py");
    // P13.3 / T1: SEO service-page enrichment (xlsx → service_page docs).
    await runPythonStep("Ingest service pages (xlsx)", "RAG_RUNTIME/scripts/ingest_service_pages.py");
    // P13.3 / T7: anonymized historical deals from orders.csv.
    await runPythonStep("Ingest historical deals (orders.csv)", "RAG_RUNTIME/scripts/ingest_orders.py");
    await runPythonStep("Ingest bridges", "RAG_RUNTIME/scripts/ingest_bridges.py");

    // Phase 3: embed + upsert to Qdrant
    await runPythonStep(
      "Build Qdrant index (full reindex)",
      "RAG_RUNTIME/scripts/build_index.py",
      ["--recreate", "--batch-size", BATCH_SIZE],
    );
  } else {
    console.log("\n[--skip-index] ingest + build_index пропущены");
  }

  // Phase 4: verify
  await runStep("Verifying SALES_RAG bundle", "verifySalesRag.mjs");
  console.log("\nSALES_RAG refresh completed successfully.");
}

main().catch((error) => {
  console.error(`\nSALES_RAG refresh failed: ${error.message}`);
  process.exit(1);
});
