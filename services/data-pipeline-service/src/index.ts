import express, { Application } from 'express';
import { ETLPipeline } from './pipelines/etl-pipeline';
import { DataValidator } from './validators/data-validator';
import { DataTransformer } from './transformers/data-transformer';
import { DataLoader } from './loaders/data-loader';
import { PipelineOrchestrator } from './orchestrators/pipeline-orchestrator';

const app: Application = express();
const PORT = process.env.PORT || 8095;

app.use(express.json({ limit: '50mb' }));

const etlPipeline = new ETLPipeline();
const dataValidator = new DataValidator();
const dataTransformer = new DataTransformer();
const dataLoader = new DataLoader();
const orchestrator = new PipelineOrchestrator();

app.post('/pipeline/execute', async (req, res) => {
  try {
    const { pipelineId, data, config } = req.body;
    const result = await orchestrator.executePipeline(pipelineId, data, config);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/pipeline/validate', async (req, res) => {
  try {
    const result = await dataValidator.validate(req.body.data, req.body.schema);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/pipeline/transform', async (req, res) => {
  try {
    const result = await dataTransformer.transform(req.body.data, req.body.rules);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/pipeline/load', async (req, res) => {
  try {
    const result = await dataLoader.load(req.body.data, req.body.destination);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/pipeline/status/:pipelineId', async (req, res) => {
  try {
    const status = await orchestrator.getPipelineStatus(req.params.pipelineId);
    res.json(status);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Data Pipeline Service running on port ${PORT}`);
});

export { app };
