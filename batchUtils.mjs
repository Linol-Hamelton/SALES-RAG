export async function batchProcess(items, batchSize, callback, delayMs = 1200) {
  const results = [];

  for (let index = 0; index < items.length; index += batchSize) {
    const batch = items.slice(index, index + batchSize);
    const batchNumber = Math.floor(index / batchSize) + 1;
    const totalBatches = Math.ceil(items.length / batchSize);
    console.log(`Processing batch ${batchNumber}/${totalBatches} (${batch.length} items)`);

    try {
      const result = await callback(batch);
      if (Array.isArray(result)) {
        results.push(...result);
      } else {
        results.push(result);
      }
    } catch (error) {
      console.error(`Batch ${batchNumber} failed: ${error.message}`);
      throw error;
    }

    if (index + batchSize < items.length) {
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  return results;
}
