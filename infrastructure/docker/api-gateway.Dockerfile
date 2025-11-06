FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
COPY tsconfig.json ./

RUN npm ci --only=production

COPY services/api-gateway ./services/api-gateway
COPY packages/shared ./packages/shared

RUN npm run build --workspace=services/api-gateway

FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache tini

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/services/api-gateway/dist ./dist
COPY --from=builder /app/packages/shared/dist ./packages/shared/dist

ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

USER node

ENTRYPOINT ["/sbin/tini", "--"]

CMD ["node", "dist/index.js"]

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"
