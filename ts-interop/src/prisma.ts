import { PrismaClient } from "@prisma/client";

// PrismaClient is instantiated once but does NOT open a connection until the
// first query is issued. Importing this module therefore never blocks
// `GET /health` or `GET /feeds/:feedType` when PostgreSQL is unavailable;
// the write path surfaces per-event errors instead of crashing.
const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient };

export const prisma: PrismaClient =
  globalForPrisma.prisma ?? new PrismaClient();

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}

/**
 * Cheap connectivity probe for PostgreSQL. Returns false (never throws) when
 * the database is down or unreachable.
 */
export async function checkPostgres(): Promise<boolean> {
  try {
    await prisma.$queryRaw`SELECT 1`;
    return true;
  } catch {
    return false;
  }
}
