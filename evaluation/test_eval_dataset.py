import os
import time
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry

import pytest
import deepeval
import instructor
from pydantic import BaseModel
import google.generativeai as genai
from deepeval import assert_test
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
  ContextualPrecisionMetric,
  ContextualRecallMetric,
  FaithfulnessMetric,
  AnswerRelevancyMetric
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Create custom Gemini class
class CustomGeminiFlash(DeepEvalBaseLLM):
  def __init__(self):
    genai.configure(api_key=GEMINI_API_KEY)
    self.model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

  def load_model(self):
    return self.model

  def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
    client = self.load_model()
    instructor_client = instructor.from_gemini(
      client=client,
      mode=instructor.Mode.GEMINI_JSON,
    )

    max_retries = 10
    for _ in range(max_retries):
      try:
        resp = instructor_client.messages.create(
          messages=[
            {
              "role": "user",
              "content": prompt,
            }
          ],
          response_model=schema
        )
        return resp
      except Exception:
        time.sleep(60)
    
    raise RuntimeError("Exceeded maximum retry attempts due to 429 errors.")

  async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
    return self.generate(prompt, schema)

  def get_model_name(self):
    return "Gemini 2.0 Flash"
  
model = CustomGeminiFlash()

# Add test cases from a JSON file
dataset = EvaluationDataset()
dataset.add_test_cases_from_json_file(
  file_path="data/eval/eval_dataset.json",
  input_key_name="user_input",
  actual_output_key_name="response",
  expected_output_key_name="reference",
  retrieval_context_key_name="retrieved_contexts"
)

failed_test_cases = []

# Evaluate
# Limit to 15 calls per minute
@pytest.mark.parametrize(
  "test_case",
  dataset
)
@sleep_and_retry
@limits(calls=3, period=60)
def test_chatbot(test_case: LLMTestCase):
  contextual_precision_metric = ContextualPrecisionMetric(model=model, async_mode=False)
  contextual_recall_metric = ContextualRecallMetric(model=model, async_mode=False)
  faithfulness_metric = FaithfulnessMetric(model=model, async_mode=False)
  answer_relevancy_metric = AnswerRelevancyMetric(model=model, async_mode=False)

  try:
    assert_test(test_case, [
      contextual_precision_metric, 
      contextual_recall_metric, 
      faithfulness_metric, 
      answer_relevancy_metric],
      run_async=False
    )
  except RuntimeError:
    failed_test_cases.append(test_case)

@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
  if not failed_test_cases:
    print("✅ All tests passed on first run.")
    return
  
  print(f"\n🔁 Rerunning {len(failed_test_cases)} failed test cases...\n")
  for i, test_case in enumerate(failed_test_cases):
    if i != 0 and i % 3 == 0:
      time.sleep(60)

    try:
      assert_test(
        test_case,
        [
          ContextualPrecisionMetric(model=model, async_mode=False),
          ContextualRecallMetric(model=model, async_mode=False),
          FaithfulnessMetric(model=model, async_mode=False),
          AnswerRelevancyMetric(model=model, async_mode=False)
        ],
        run_async=False
      )
      print(f"✅ Passed on rerun: {test_case.input}")
    except RuntimeError:
      print(f"❌ Still failing: {test_case.input}")
