import fs from 'node:fs/promises';
import 'dotenv/config';
import {
  Document,
  MetadataMode,
  NodeWithScore,
  Ollama,
  Settings,
  VectorStoreIndex,
} from 'llamaindex';

const ollama = new Ollama({
  model: process.env.OLLAMA_MODEL ?? 'llama3.1',
  config: {
    temperature: parseFloat(process.env.OLLAMA_TEMPERATURE ?? '0.1'),
  },
});
Settings.llm = ollama;
Settings.embedModel = ollama;

export async function main() {
  const path = './src/epam-rag-from-scratch/story.txt';
  const story = await fs.readFile(path, 'utf-8');
  const document = new Document({text: story, id_: path});

  const pathBlackHoodStory = './src/epam-rag-from-scratch/black-hood-story.txt';
  const blackHoodStory = await fs.readFile(pathBlackHoodStory, 'utf-8');
  const documentBlackHoodStory = new Document({
    text: blackHoodStory,
    id_: pathBlackHoodStory,
  });

  const index = await VectorStoreIndex.fromDocuments([
    document,
    documentBlackHoodStory,
  ]);

  const queryEngine = index.asQueryEngine();
  const {message, sourceNodes} = await queryEngine.query({
    query: `
      Your goal is to answer the following question: "Who does iPhone belong to?"
      Also provide the statements on which the answer is based.
    `,
  });

  console.log(message.content);

  if (sourceNodes) {
    sourceNodes.forEach((source: NodeWithScore, index: number) => {
      console.log(
        `\n${index}: Score: ${source.score} - ${source.node
          .getContent(MetadataMode.NONE)
          .substring(0, 50)}...\n`
      );
    });
  }

  const stream = await queryEngine.query({
    query: `
      Your goal is to answer the following question: "Did Black Hat meet a king one day?"
      Also provide the statements on which the answer is based.
    `,
    stream: true,
  });
  for await (const chunk of stream) {
    process.stdout.write(chunk.response);
  }
}
