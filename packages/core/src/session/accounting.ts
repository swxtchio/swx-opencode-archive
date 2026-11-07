export * as SessionAccounting from "./accounting"

import { LLMEvent, type Model, type ProviderMetadata, type Usage } from "@opencode-ai/llm"
import { DateTime, Effect } from "effect"
import { Catalog } from "../catalog"
import { Config } from "../config"
import { EventV2 } from "../event"
import { ModelV2 } from "../model"
import { ProviderV2 } from "../provider"
import { SessionSchema } from "./schema"
import { SessionAccountingEvent } from "@opencode-ai/schema/session-accounting-event"
import { SessionMessage } from "./message"

export type Operation = "turn" | "title" | "compaction"
export type Tokens = SessionAccountingEvent.Tokens

export type ProviderMeasurement = {
  readonly usage?: Usage
  readonly responseModelID?: string
  readonly providerMetadata?: ProviderMetadata
}

export type Measurement = {
  readonly cost: number
  readonly tokens: Tokens
}

export const capture = () => {
  let step: ProviderMeasurement | undefined
  let finish: ProviderMeasurement | undefined

  const captureEvent = (event: LLMEvent) => {
    if (LLMEvent.is.stepFinish(event)) {
      if (step) return
      step = {
        ...(event.usage === undefined ? {} : { usage: event.usage }),
        ...(event.responseModelID === undefined ? {} : { responseModelID: event.responseModelID }),
        ...(event.providerMetadata === undefined ? {} : { providerMetadata: event.providerMetadata }),
      }
      return
    }
    if (!LLMEvent.is.finish(event)) return
    finish = {
      ...(event.usage === undefined ? {} : { usage: event.usage }),
      ...(event.responseModelID === undefined ? {} : { responseModelID: event.responseModelID }),
      ...(event.providerMetadata === undefined ? {} : { providerMetadata: event.providerMetadata }),
    }
  }

  const value = () => {
    if (!step) return finish
    if (!finish) return step
    return {
      ...(step.usage ?? finish.usage ? { usage: step.usage ?? finish.usage } : {}),
      ...(step.responseModelID ?? finish.responseModelID
        ? { responseModelID: step.responseModelID ?? finish.responseModelID }
        : {}),
      ...(step.providerMetadata ?? finish.providerMetadata
        ? { providerMetadata: step.providerMetadata ?? finish.providerMetadata }
        : {}),
    }
  }

  return { capture: captureEvent, value }
}

type Dependencies = {
  readonly events: EventV2.Interface
  readonly catalog: Catalog.Interface
  readonly config: readonly Config.Entry[]
}

type Input = {
  readonly sessionID: SessionSchema.ID
  readonly requestID: string
  readonly operation: Operation
  readonly model: Model
  readonly assistantMessageID?: SessionMessage.ID
  readonly usage?: Usage
  readonly responseModelID?: string
  readonly providerMetadata?: ProviderMetadata
}

export const make = (dependencies: Dependencies) => {
  const record = Effect.fn("SessionAccounting.record")(function* (input: Input) {
    if (input.usage === undefined && input.responseModelID === undefined) return
    const modelID = ModelV2.ID.make(input.model.id)
    const providerID = ProviderV2.ID.make(input.model.provider)
    const requestedModel = yield* dependencies.catalog.model.get(providerID, modelID).pipe(
      Effect.map((model) => model ?? ModelV2.Info.empty(providerID, modelID)),
    )
    const servedModel =
      input.responseModelID === undefined || input.responseModelID === input.model.id
        ? undefined
        : yield* dependencies.catalog.model.get(providerID, ModelV2.ID.make(input.responseModelID))
    const pricedModel =
      servedModel &&
      (hasKnownPrice(servedModel) || hasConfiguredPrice(dependencies.config, providerID, servedModel.id))
        ? servedModel
        : requestedModel
    const measurement = input.usage === undefined ? undefined : usage(pricedModel, input.usage, input.providerMetadata)

    yield* dependencies.events.publish(SessionAccountingEvent.Recorded, {
      sessionID: input.sessionID,
      timestamp: yield* DateTime.now,
      requestID: input.requestID,
      operation: input.operation,
      requestedModel: { id: modelID, providerID },
      assistantMessageID: input.assistantMessageID,
      responseModelID: input.responseModelID,
      ...(measurement === undefined ? {} : measurement),
    })
    return measurement
  })

  return { record }
}

function hasKnownPrice(model: ModelV2.Info) {
  return model.cost.some((cost) =>
    [cost.input, cost.output, cost.cache.read, cost.cache.write].some((rate) => rate > 0),
  )
}

function hasConfiguredPrice(config: readonly Config.Entry[], providerID: ProviderV2.ID, modelID: ModelV2.ID) {
  return config.some(
    (entry) =>
      entry.type === "document" && entry.info.providers?.[providerID]?.models?.[modelID]?.cost !== undefined,
  )
}

function usage(model: ModelV2.Info, input: Usage, providerMetadata: ProviderMetadata | undefined): Measurement {
  const safe = (value: number) => (Number.isFinite(value) ? Math.max(0, value) : 0)
  const finite = (value: number | undefined) => (Number.isFinite(value) ? (value ?? 0) : 0)
  const record = (value: unknown): Record<string, unknown> =>
    typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
  const metadata = record(providerMetadata)
  const cacheWriteInputTokens = safe(
    Number(
      input.cacheWriteInputTokens ??
        record(metadata["anthropic"])["cacheCreationInputTokens"] ??
        record(metadata["vertex"])["cacheCreationInputTokens"] ??
        record(record(metadata["bedrock"])["usage"])["cacheWriteInputTokens"] ??
        record(record(metadata["venice"])["usage"])["cacheCreationInputTokens"] ??
        0,
    ),
  )
  const cacheReadInputTokens = safe(input.cacheReadInputTokens ?? 0)
  const inputTokens = safe(input.inputTokens ?? 0)
  const outputTokens = safe(input.outputTokens ?? 0)
  const reasoningTokens = safe(input.reasoningTokens ?? 0)
  const tokens: Tokens = {
    input: safe(inputTokens - cacheReadInputTokens - cacheWriteInputTokens),
    output: safe(outputTokens - reasoningTokens),
    reasoning: reasoningTokens,
    cache: { read: cacheReadInputTokens, write: cacheWriteInputTokens },
  }
  const costInfo =
    model.cost
      .filter((item) => item.tier?.type === "context" && inputTokens > item.tier.size)
      .toSorted((a, b) => (b.tier?.size ?? 0) - (a.tier?.size ?? 0))[0] ??
    model.cost.find((item) => item.tier === undefined) ??
    model.cost[0]
  const nanoAiu = record(metadata["copilot"])["totalNanoAiu"]
  const cost =
    typeof nanoAiu === "number" && Number.isFinite(nanoAiu) && nanoAiu >= 0
      ? safe(nanoAiu / 100_000_000_000)
      : safe(
          (tokens.input * finite(costInfo?.input) +
            tokens.output * finite(costInfo?.output) +
            tokens.reasoning * finite(costInfo?.output) +
            tokens.cache.read * finite(costInfo?.cache.read) +
            tokens.cache.write * finite(costInfo?.cache.write)) /
            1_000_000,
        )
  return { cost, tokens }
}
