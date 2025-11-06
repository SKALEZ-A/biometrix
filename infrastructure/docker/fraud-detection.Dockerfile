FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
COPY tsconfig.json ./

RUN npm ci --only=production

COPY services/fraud-detection-service ./services/fraud-detection-service
COPY packages/shared ./packages/shared
COPY ml-models ./ml-models

RUN npm run build:fraud-detection-service

FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache dumb-init python3 py3-pip

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/services/fraud-detection-service/dist ./dist
COPY --from=builder /app/packages/shared/dist ./packages/shared/dist
COPY --from=builder /app/ml-models ./ml-models

RUN pip3 install --no-cache-dir numpy pandas scikit-learn tensorflow xgboost

ENV NODE_ENV=production
ENV PORT=3000
ENV ML_MODEL_PATH=/app/ml-models

EXPOSE 3000

USER node

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]
