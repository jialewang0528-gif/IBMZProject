import os
import re

from llama_cpp import Llama
from rank_bm25 import BM25Okapi

from loan_demo.pdf_utils import extract_text_from_pdf

# Flip this flag to switch models:
#   True  -> granite-4.0-h-350m-base (base model, raw text completion)
#   False -> granite-3.3-2b-instruct (instruct model, chat completion)
USE_BASE_MODEL = True

INSTRUCT_MODEL_PATH = "/home/ibm-user/granite-3.3-2b-instruct-be.Q4_K_M.gguf"
BASE_MODEL_PATH = "/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf"

MODEL_PATH = BASE_MODEL_PATH if USE_BASE_MODEL else INSTRUCT_MODEL_PATH

class LocalModel:

    def __init__(self):

        self.model = Llama(
            model_path=MODEL_PATH,
            n_ctx=16384,
            n_batch=512,
            n_threads=8,
            verbose=False,
        )

    def generate(self, prompt, max_tokens=250, stop=None):
        
        if USE_BASE_MODEL:
            # Base models have no chat template, so use raw text completion;
            # chat completion returns empty output for them.
            stream = self.model.create_completion(
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
            )
        else:
            stream = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
            )
    
        full_text = ""
    
        for chunk in stream:
            choice = chunk["choices"][0]
            if USE_BASE_MODEL:
                token = choice.get("text", "")
            else:
                token = choice.get("delta", {}).get("content", "")
    
            if token:
                print(token, end="", flush=True)
                full_text += token
    
        print()
        
        return full_text


def load_model():
    return LocalModel()


def load_policy_docs(folder_path: str):
    """
    Load all policy PDFs.
    """

    docs = []

    for file in os.listdir(folder_path):

        # skip macOS AppleDouble resource-fork sidecars (._filename)
        if file.startswith("._"):
            continue

        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder_path, file)

        docs.append(
            {
                "title": file,
                "content": extract_text_from_pdf(path),
            }
        )

    return docs


def build_policy_context(policy_docs):
    """
    Concatenate policy text.
    """

    return "\n\n".join(
        f"{doc['title']}\n{doc['content']}"
        for doc in policy_docs
    )


def build_policy_index(policy_docs, chunk_size=250):
    policy_context = build_policy_context(policy_docs)

    chunks = [
        policy_context[i:i + chunk_size]
        for i in range(0, len(policy_context), chunk_size)
    ]

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25, chunks


def retrieve_policy_context(
    bm25,
    chunks,
    applicant,
    k=1,
):
    query = f"""
    fico {applicant.get('FICO_Score', '')}
    debt to income ratio {applicant.get('Debt_To_Income_Ratio', '')}
    mortgage approval
    income verification
    credit history
    """

    results = bm25.get_top_n(
        query.lower().split(),
        chunks,
        n=k,
    )

    return "\n\n".join(results)


def summarize_applicant(applicant: dict):
    return f"""
Applicant Summary
FICO={applicant.get("FICO_Score")}
DTI={applicant.get("Debt_To_Income_Ratio")}
Income={applicant.get("Gross_Annual_Income(in dollars)")}
LoanAmount={applicant.get("Loan_Amount(in dollars)")}
LoanType={applicant.get("Loan_Type")}
Purpose={applicant.get("Loan_Purpose")}
CreditUtilization={applicant.get("Credit_Utilization_Ratio")}
Age={applicant.get("Applicant_Age")}
""".strip()


def compress_applicant_documents(
    bank_text: str,
    paystub_text: str,
    w2_text: str,
):
    """
    Compress large synthetic mortgage documents into a
    compact underwriting evidence summary.
    """

    summary = []

    # BANK STATEMENT
    bank_values = re.findall(
        r"\d{1,3}(?:,\d{3})*\.\d{2}",
        bank_text,
    )

    if len(bank_values) >= 4:

        opening = bank_values[0]
        closing = bank_values[1]
        credits = bank_values[2]
        debits = bank_values[3]

        try:
            cashflow = (
                "positive"
                if float(credits.replace(",", ""))
                >
                float(debits.replace(",", ""))
                else "negative"
            )
        except Exception:
            cashflow = "unknown"

        summary.append(
            f"Bank: "
            f"opening={opening}, "
            f"closing={closing}, "
            f"credits={credits}, "
            f"debits={debits}, "
            f"cashflow={cashflow}"
        )

    # PAYSTUB
    pay_values = re.findall(
        r"\d{1,3}(?:,\d{3})*\.\d{2}",
        paystub_text,
    )
    
    if len(pay_values) >= 6:
    
        latest = pay_values[-6:]

        summary.append(
            f"Paystub: "
            f"ytd_gross={latest[0]}, "
            f"ytd_deductions={latest[1]},"
            f"ytd_net={latest[2]}, "
            f"gross_pay={latest[3]}, "
            f"net_pay={latest[5]}"
        )

    # W2
    w2_values = re.findall(
        r"\d{1,3}(?:,\d{3})*\.\d{2}",
        w2_text,
    )

    if len(w2_values) >= 8:

        latest_wages = w2_values[-8]
        latest_federal_tax = w2_values[-7]

        summary.append(
            f"W2 Latest: wages={latest_wages}, "
            f"federal_tax={latest_federal_tax}"
        )

    else:

        summary.append(
            f"W2 Recent:\n{w2_text[-300:]}"
        )

    return "\n\n".join(summary)


def extract_document_features(
    model,
    document_text: str,
):
    prompt = f"""
You are reviewing mortgage underwriting documents.

Guidelines:
- Positive cashflow indicates Positive cash flow stability.
- A substantially higher closing balance than opening balance indicates High financial stability.
- Consistent paystub and W2 income indicates High income consistency.
- Presence of normal mortgage documentation indicates Compliant status.

Return EXACTLY four lines:

Financial Stability: [Low/Medium/High]
Cash Flow Stability: [Positive/Neutral/Negative]
Income Consistency: [Low/Medium/High]
Compliance Status: [Compliant/Review Needed]

Evidence:
{document_text}

Answer:
"""

    return model.generate(prompt, 50)


def generate_answer(
    model,
    context: str,
    question: str,
):
    """
    Generic RAG-style Q&A.
    """

    if USE_BASE_MODEL:
        # Small base models copy the "Insufficient information" fallback and
        # then keep generating (hallucinating). Use a one-shot example, no
        # fallback line, and stop sequences to end after a single answer.
        prompt = f"""You are a mortgage approval assistant. Answer the question in 2-3 sentences using ONLY the facts in the context.

Context:
Applicant: Income $90,000; Credit score 780; Loan amount $250,000
Risk: Low risk
Document Analysis:
Financial Stability: High
Cash Flow Stability: Positive
Income Consistency: High
Compliance Status: Compliant
Recommendation: Approve

Question: Why was this applicant approved?
Answer: The applicant has a high credit score and strong income relative to the loan amount. Documents show positive cash flow, consistent income and compliance, and the risk model predicts low risk.

Context:
{context}

Question: {question}
Answer:"""

        return model.generate(
            prompt,
            150,
            stop=["\n\n", "\nQuestion:", "\nContext:"],
        )

    prompt = f"""
You are a mortgage approval assistant.

Use ONLY the provided context.

Question:
{question}

Context:
{context}

Answer in 2-4 concise sentences.

If the answer is not present in the context, respond:

Insufficient information available.

Answer:
"""

    return model.generate(prompt, 150)


def generate_recommendation(
    model,
    applicant,
    risk_result,
    document_features,
    bm25,
    policy_chunks,
):
    """
    Final loan recommendation.
    """

    policy_context = retrieve_policy_context(
        bm25,
        policy_chunks,
        applicant,
    )

    if USE_BASE_MODEL:
        # A base model ignores "maximum 20 words" and keeps writing until it
        # hits max_tokens. Show the exact 2-line format with a short example
        # and stop at the first blank line / next section instead.
        prompt = f"""You are a mortgage approval advisor. Use ONLY the supplied information.

Applicant:
Credit score 780, DTI 22%, income $90,000, loan $250,000
Risk Prediction:
Low risk
Document Analysis:
Financial Stability: High
Cash Flow Stability: Positive
Income Consistency: High
Compliance Status: Compliant
Policy Guidance:
Applicants with credit score above 740 and DTI below 36% qualify for approval.
Recommendation: Approve
Main Reason: High credit score, low DTI and stable documented income meet policy.

Applicant:
{summarize_applicant(applicant)}
Risk Prediction:
{risk_result}
Document Analysis:
{document_features}
Policy Guidance:
{policy_context}
Recommendation:"""

        # The prompt ends with the label, so print it here for the stream.
        print("Recommendation:", end="", flush=True)
        text = model.generate(
            prompt,
            60,
            stop=["\n\n", "\nApplicant:"],
        )
        return "Recommendation:" + text

    prompt = f"""
You are a mortgage approval advisor.

Use ONLY the supplied information.

Applicant:
{summarize_applicant(applicant)}

Risk Prediction:
{risk_result}

Document Analysis:
{document_features}

Policy Guidance:
{policy_context}

Answer using EXACTLY this format:

Recommendation: <Approve, Reject, Review>
Main Reason: <maximum 20 words>

Now generate the recommendation for this applicant.

Answer:
"""

    return model.generate(prompt, 75)
    