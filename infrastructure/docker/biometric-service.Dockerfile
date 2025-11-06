FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
COPY tsconfig.json ./

RUN npm ci --only=production

COPY services/biometric-service ./services/biometric-service
COPY packages/shared ./packages/shared

RUN npm run build:biometric-service

FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache dumb-init

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/services/biometric-service/dist ./dist
COPY --from=builder /app/packages/shared/dist ./packages/shared/dist

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

USER node

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]
