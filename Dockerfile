# Multi-stage build. The runtime image carries the built backend, the built
# frontend (served as static files from the same origin) and the Prisma engine.

FROM node:24-slim AS builder
WORKDIR /app

# OpenSSL must be present at generate time: without it Prisma cannot detect the
# libssl version, guesses debian-openssl-1.1.x, and the runtime then fails to
# locate a query engine. schema.prisma pins both debian targets as a belt to
# this braces.
RUN apt-get update && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

RUN corepack enable && corepack prepare pnpm@10.15.0 --activate

COPY package.json pnpm-workspace.yaml pnpm-lock.yaml .npmrc tsconfig.base.json ./
COPY shared/package.json  shared/
COPY backend/package.json backend/
COPY frontend/package.json frontend/
RUN pnpm install --frozen-lockfile

COPY shared/   shared/
COPY backend/  backend/
COPY frontend/ frontend/
COPY data/     data/

RUN pnpm --filter @ipp/shared build \
    && pnpm --filter @ipp/backend exec prisma generate \
    && pnpm --filter @ipp/frontend build \
    && pnpm --filter @ipp/backend build

# Strip dev dependencies from what we ship.
RUN pnpm --filter @ipp/backend --prod deploy /tmp/backend

# ---------------------------------------------------------------------------

FROM node:24-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends openssl \
    && rm -rf /var/lib/apt/lists/*

ENV NODE_ENV=production

COPY --from=builder /tmp/backend/node_modules ./node_modules
COPY --from=builder /app/backend/dist         ./dist
COPY --from=builder /app/backend/public       ./public
COPY --from=builder /app/backend/prisma       ./prisma

# `pnpm deploy` resolves dependencies from the lockfile and so misses the
# client that `prisma generate` wrote into the store at build time -- the
# runtime then dies on "Cannot find module '.prisma/client/default'". Copy the
# generated client over the deployed tree.
COPY --from=builder /app/node_modules/.pnpm/@prisma+client*/node_modules/.prisma \
     ./node_modules/.prisma

# The corpus ships in the image so a fresh deployment can index itself without
# reaching back to a developer machine. It is 20 MB of text.
COPY --from=builder /app/data ./data

EXPOSE 3000
USER node

CMD ["node", "dist/server.js"]
