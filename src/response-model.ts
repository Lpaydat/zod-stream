import OpenAI from "openai";
import { z } from "zod";
import zodToJsonSchema from "zod-to-json-schema";

import { MODE } from "@/constants/modes";
import {
  OAIBuildFunctionParams,
  OAIBuildJsonModeParams,
  OAIBuildJsonSchemaParams,
  OAIBuildMessageBasedParams,
  OAIBuildToolFunctionParams,
} from "@/oai/params";

import {
  Mode,
  ModeParamsReturnType,
  ResponseModel,
  ResponseSchema,
} from "./types";

function buildDefinition<T>(
  name: string,
  schema: T,
  description: string,
  isZodSchema: boolean
) {
  const safeName = name.replace(/[^a-zA-Z0-9]/g, "_").replace(/\s/g, "_");

  let definition;
  if (isZodSchema) {
    const { definitions } = zodToJsonSchema(schema as z.AnyZodObject, {
      name: safeName,
      errorMessages: true,
    });

    if (!definitions || !definitions?.[safeName]) {
      console.warn(
        "Could not extract json schema definitions from your schema",
        schema
      );
      throw new Error(
        "Could not extract json schema definitions from your schema"
      );
    }

    definition = {
      name: safeName,
      description,
      ...definitions[safeName],
    };
  } else {
    definition = {
      name: safeName,
      description,
      ...schema,
    };
  }

  return definition;
}

function buildParams<
  P extends OpenAI.ChatCompletionCreateParams,
  M extends Mode
>(definition: any, mode: M, params: P): ModeParamsReturnType<P, M> {
  switch (mode) {
    case MODE.FUNCTIONS:
      return OAIBuildFunctionParams<P>(
        definition,
        params
      ) as ModeParamsReturnType<P, M>;
    case MODE.TOOLS:
      return OAIBuildToolFunctionParams<P>(
        definition,
        params
      ) as ModeParamsReturnType<P, M>;
    case MODE.JSON:
      return OAIBuildJsonModeParams<P>(
        definition,
        params
      ) as ModeParamsReturnType<P, M>;
    case MODE.JSON_SCHEMA:
      return OAIBuildJsonSchemaParams<P>(
        definition,
        params
      ) as ModeParamsReturnType<P, M>;
    case MODE.MD_JSON:
    default:
      return OAIBuildMessageBasedParams<P>(
        definition,
        params
      ) as ModeParamsReturnType<P, M>;
  }
}

// Define a union type for response_model
type ResponseModelUnion<T extends z.AnyZodObject> =
  | ResponseModel<T>
  | ResponseSchema;

export function withResponseModel<
  T extends z.AnyZodObject,
  M extends Mode,
  P extends OpenAI.ChatCompletionCreateParams
>({
  response_model,
  mode,
  params,
}: {
  response_model: ResponseModelUnion<T>;
  mode: M;
  params: P;
}): ModeParamsReturnType<P, M> {
  const { name, schema, description = "" } = response_model;

  let definition;
  if (schema instanceof z.ZodType) {
    definition = buildDefinition(name, schema, description, true);
  } else {
    definition = buildDefinition(name, schema, description, false);
  }

  return buildParams(definition, mode, params);
}
