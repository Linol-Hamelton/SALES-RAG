import { spawn } from "child_process";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const ROOT_DIR = dirname(fileURLToPath(import.meta.url));

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

async function main() {
  await runStep("Refreshing raw CSV from Bitrix24", "RAG_DATA/generateRagData.mjs");
  await runStep("Repairing orders direction", "RAG_DATA/repairOrdersDirection.mjs");
  await runStep("Rebuilding analytics outputs", "RAG_ANALYTICS/runAll.mjs");
  await runStep("Verifying SALES_RAG bundle", "verifySalesRag.mjs");
  console.log("\nSALES_RAG refresh completed successfully.");
}

main().catch((error) => {
  console.error(`\nSALES_RAG refresh failed: ${error.message}`);
  process.exit(1);
});
