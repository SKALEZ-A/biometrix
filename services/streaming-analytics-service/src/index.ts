import express, { Application } from 'express';
import { StreamProcessor } from './processors/stream-processor';
import { WindowAggregator } from './aggregators/window-aggregator';
import { EventTimeProcessor } from './processors/event-time-processor';
import { StateManager } from './state/state-manager';

const app: Application = express();
const PORT = process.env.PORT || 8090;

app.use(express.json());

const streamProcessor = new StreamProcessor();
const windowAggregator = new WindowAggregator();
const eventTimeProcessor = new EventTimeProcessor();
const stateManager = new StateManager();

app.post('/stream/process', async (req, res) => {
  try {
    const result = await streamProcessor.process(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/stream/aggregate', async (req, res) => {
  try {
    const result = await windowAggregator.aggregate(req.body);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Streaming Analytics Service running on port ${PORT}`);
});
