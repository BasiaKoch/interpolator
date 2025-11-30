###############################################
# STAGE 1 — Install dependencies
###############################################
FROM node:20-alpine AS deps

WORKDIR /app

# Copy only package files for efficient caching
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci


###############################################
# STAGE 2 — Build Next.js application
###############################################
FROM node:20-alpine AS builder

WORKDIR /app

# Copy installed node_modules
COPY --from=deps /app/node_modules ./node_modules

# Copy the full project
COPY . .

# Disable telemetry & set production mode
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production

# Build the Next.js app
RUN npm run build


###############################################
# STAGE 3 — Production runner
###############################################
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Install wget for health checks
RUN apk add --no-cache wget

# Create secure non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy Next.js build output
COPY --from=builder /app/.next  ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Ensure permissions belong to the non-root user
RUN chown -R nextjs:nodejs /app

USER nextjs

EXPOSE ${PORT}

###############################################
# Health check endpoint (optional)
###############################################
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:${PORT} >/dev/null || exit 1

###############################################
# Start the Next.js server
###############################################
CMD ["npm", "start"]

