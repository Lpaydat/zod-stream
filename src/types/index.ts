import OpenAI from "openai";
import { z } from "zod";
import { JsonSchema7Type } from "zod-to-json-schema";

import { MODE } from "@/constants/modes";

export type ActivePath = (string | number | undefined)[];
export type CompletedPaths = ActivePath[];

export type CompletionMeta = {
  _activePath: ActivePath;
  _completedPaths: CompletedPaths;
  _isValid: boolean;
};

export type LogLevel = "debug" | "info" | "warn" | "error";

export type ClientConfig = {
  debug?: boolean;
};

export type ParseParams = {
  name: string;
  description?: string;
} & JsonSchema7Type;

export type Mode = keyof typeof MODE;

export type ResponseModel<T extends z.AnyZodObject> = {
  schema: T;
  name: string;
  description?: string;
};

export type ResponseSchema = {
  schema: object | JsonSchema7Type;
  name: string;
  description?: string;
};

export type ZodStreamCompletionParams<T extends z.AnyZodObject> = {
  response_model: { schema: T };
  data?: Record<string, unknown>;
  completionPromise: (
    data?: Record<string, unknown>
  ) => Promise<ReadableStream<Uint8Array>>;
};

export type InferStreamType<T extends OpenAI.ChatCompletionCreateParams> =
  T extends {
    stream: true;
  }
    ? OpenAI.ChatCompletionCreateParamsStreaming
    : OpenAI.ChatCompletionCreateParamsNonStreaming;

export type FunctionParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams
> = T & {
  function_call: OpenAI.ChatCompletionFunctionCallOption;
  functions: OpenAI.FunctionDefinition[];
};

export type ToolFunctionParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams
> = T & {
  tool_choice: OpenAI.ChatCompletionToolChoiceOption;
  tools: OpenAI.ChatCompletionTool[];
};

export type MessageBasedParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams
> = T;

export type JsonModeParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams
> = T & {
  response_format: { type: "json_object" };
  messages: OpenAI.ChatCompletionMessageParam[];
};

export type JsonSchemaParamsReturnType<
  T extends Omit<OpenAI.ChatCompletionCreateParams, "response_format">
> = T & {
  response_format: {
    type: "json_object";
    schema: JsonSchema7Type;
  };
  messages: OpenAI.ChatCompletionMessageParam[];
};

export type ModeParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams,
  M extends Mode
> = ReturnType<typeof getModeParamsReturnType<T, M>>;

// Helper function to determine ModeParamsReturnType
function getModeParamsReturnType<
  T extends OpenAI.ChatCompletionCreateParams,
  M extends Mode
>(params: T, mode: M) {
  const baseParams = { ...params };

  switch (mode) {
    case MODE.FUNCTIONS:
      return {
        ...baseParams,
        function_call: {} as OpenAI.ChatCompletionFunctionCallOption,
        functions: [] as OpenAI.FunctionDefinition[],
      };
    case MODE.TOOLS:
      return {
        ...baseParams,
        tool_choice: {} as OpenAI.ChatCompletionToolChoiceOption,
        tools: [] as OpenAI.ChatCompletionTool[],
      };
    case MODE.JSON:
      return {
        ...baseParams,
        response_format: { type: "json_object" },
        messages: [] as OpenAI.ChatCompletionMessageParam[],
      };
    case MODE.JSON_SCHEMA:
      return {
        ...baseParams,
        response_format: {
          type: "json_object",
          schema: {} as JsonSchema7Type,
        },
        messages: [] as OpenAI.ChatCompletionMessageParam[],
      };
    case MODE.MD_JSON:
    default:
      return baseParams;
  }
}
