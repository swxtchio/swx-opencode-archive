export * as SessionAccountingEvent from "./session-accounting-event"

import { Schema } from "effect"
import { Event } from "./event"
import { Model } from "./model"
import { optional, DateTimeUtcFromMillis } from "./schema"
import { SessionID } from "./session-id"
import { SessionMessage } from "./session-message"

export const Tokens = Schema.Struct({
  input: Schema.Finite,
  output: Schema.Finite,
  reasoning: Schema.Finite,
  cache: Schema.Struct({ read: Schema.Finite, write: Schema.Finite }),
}).annotate({ identifier: "Session.Accounting.Tokens" })
export interface Tokens extends Schema.Schema.Type<typeof Tokens> {}

// The internal accounting ledger stays out of the client-facing event manifest.
export const Recorded = Event.define({
  type: "session.next.accounting.recorded",
  durable: { aggregate: "sessionID", version: 1 },
  schema: {
    sessionID: SessionID,
    timestamp: DateTimeUtcFromMillis,
    requestID: Schema.String,
    operation: Schema.Literals(["turn", "title", "compaction"]),
    requestedModel: Model.Ref,
    assistantMessageID: SessionMessage.ID.pipe(optional),
    responseModelID: Schema.String.pipe(optional),
    cost: Schema.Finite.pipe(optional),
    tokens: Tokens.pipe(optional),
  },
})
export type Recorded = typeof Recorded.Type

export const Definitions = Event.inventory(Recorded)
