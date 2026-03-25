"""
Production-ready RAG pipeline using OpenAI + Pinecone (no LangChain).

Features
- Serverless Pinecone index bootstrap
- Batch upsert of vectors with metadata
- OpenAI embedding generation (text-embedding-3-large)
- Semantic retrieval
- Context building with source citations
- Final RAG answer generation (gpt-4o-mini)

Environment variables required
- OPENAI_API_KEY
- PINECONE_API_KEY

Example usage
1) Ingest chunks into Pinecone:
   python rag_pinecone.py ingest --input data/chunks.json

2) Ask a question with RAG:
   python rag_pinecone.py ask --question "What is student eligibility?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

try:
	from sentence_transformers import SentenceTransformer
except ImportError:
	SentenceTransformer = None


# ---------------------------
# Configuration
# ---------------------------

INDEX_NAME = "proed-chatbot"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
INDEX_METRIC = "cosine"
INDEX_CLOUD = "aws"
INDEX_REGION = "us-east-1"
UPSERT_BATCH_SIZE = 100
DEFAULT_TOP_K = 10
_ST_MODEL_CACHE: Dict[str, Any] = {}
LLM_STATE_PATH = os.getenv("LLM_STATE_PATH", "data/rag_llm_state.json")
LLM_MIN_INTERVAL_SECONDS = int(os.getenv("LLM_MIN_INTERVAL_SECONDS", "45"))
LLM_REQUEST_DELAY_SECONDS = float(os.getenv("LLM_REQUEST_DELAY_SECONDS", "3"))
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "1800"))


@dataclass
class ChunkRecord:
	content: str
	section_ref: str
	source_type: str
	authority_level: str
	source_url: str
	chunk_id: Optional[str] = None


def setup_logging(verbose: bool = False) -> None:
	level = logging.DEBUG if verbose else logging.INFO
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(message)s",
	)


def get_required_env(name: str) -> str:
	value = os.getenv(name, "").strip()
	if not value:
		raise EnvironmentError(f"Missing required environment variable: {name}")
	return value


def ensure_parent_dir(file_path: str) -> None:
	parent = os.path.dirname(file_path)
	if parent:
		os.makedirs(parent, exist_ok=True)


def load_llm_state(path: str = LLM_STATE_PATH) -> Dict[str, Any]:
	if not os.path.exists(path):
		return {"last_llm_call_ts": 0.0, "query_cache": {}}
	with open(path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		return {"last_llm_call_ts": 0.0, "query_cache": {}}
	data.setdefault("last_llm_call_ts", 0.0)
	data.setdefault("query_cache", {})
	return data


def save_llm_state(state: Dict[str, Any], path: str = LLM_STATE_PATH) -> None:
	ensure_parent_dir(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(state, f, indent=2, ensure_ascii=False)


def normalize_query_for_cache(query: str) -> str:
	return " ".join((query or "").lower().split())


def build_retrieval_only_answer(matches: List[Dict[str, Any]]) -> str:
	if not matches:
		return "No matching sources found."

	lines = ["LLM response is rate-limited. Retrieval-only result:"]
	for i, item in enumerate(matches[:5], start=1):
		md = item.get("metadata", {})
		section = md.get("section_ref", "")
		content = (md.get("content", "") or "").strip()
		snippet = content[:280] + ("..." if len(content) > 280 else "")
		lines.append(f"[SOURCE {i}] Section: {section}")
		lines.append(f"{snippet}")

	return "\n".join(lines)


def load_dotenv(dotenv_path: str = ".env") -> None:
	if not os.path.exists(dotenv_path):
		return

	with open(dotenv_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			k, v = line.split("=", 1)
			k = k.strip()
			v = v.strip().strip('"').strip("'")
			if k and k not in os.environ:
				os.environ[k] = v


def build_openai_client() -> OpenAI:
	api_key = get_required_env("OPENAI_API_KEY")
	base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
	return OpenAI(api_key=api_key, base_url=base_url)


def build_chat_client_and_model() -> tuple[OpenAI, str]:
	provider = os.getenv("CHAT_PROVIDER", "").strip().lower()
	chat_model = CHAT_MODEL

	if not provider:
		if chat_model.lower().startswith("gemini") and os.getenv("GEMINI_API_KEY"):
			provider = "gemini"
		else:
			provider = "openai"

	if provider == "gemini":
		api_key = get_required_env("GEMINI_API_KEY")
		base_url = os.getenv(
			"GEMINI_BASE_URL",
			"https://generativelanguage.googleapis.com/v1beta/openai/",
		)
		return OpenAI(api_key=api_key, base_url=base_url), chat_model

	return build_openai_client(), chat_model


def build_pinecone_client() -> Pinecone:
	api_key = get_required_env("PINECONE_API_KEY")
	return Pinecone(api_key=api_key)


def ensure_index(pc: Pinecone, index_name: str = INDEX_NAME, dimension: int = 3072) -> None:
	listed = pc.list_indexes()
	if hasattr(listed, "names"):
		existing = set(listed.names())
	else:
		existing = {
			(idx.get("name") if isinstance(idx, dict) else getattr(idx, "name", None))
			for idx in listed
		}
		existing.discard(None)

	if index_name in existing:
		desc = pc.describe_index(index_name)
		idx_dim = getattr(desc, "dimension", None)
		if idx_dim is None and isinstance(desc, dict):
			idx_dim = desc.get("dimension")
		if idx_dim is not None and int(idx_dim) != int(dimension):
			raise ValueError(
				f"Index '{index_name}' dimension mismatch: existing={idx_dim}, expected={dimension}. "
				"Use a different index name or recreate index with correct dimension."
			)
		logging.info("Pinecone index exists: %s", index_name)
		return

	logging.info("Creating Pinecone index: %s", index_name)
	pc.create_index(
		name=index_name,
		dimension=dimension,
		metric=INDEX_METRIC,
		spec=ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
	)

	# Wait for readiness
	for _ in range(60):
		desc = pc.describe_index(index_name)
		status_obj = getattr(desc, "status", None)
		if isinstance(status_obj, dict):
			status = bool(status_obj.get("ready", False))
		else:
			status = bool(getattr(status_obj, "ready", False))
		if status:
			logging.info("Index is ready: %s", index_name)
			return
		time.sleep(2)

	raise TimeoutError(f"Index creation timed out for: {index_name}")


def get_index(pc: Pinecone, index_name: str = INDEX_NAME):
	return pc.Index(index_name)


def is_sentence_transformer_model(model: str) -> bool:
	return model.strip().lower().startswith("sentence-transformers/")


def get_sentence_transformer(model: str):
	if SentenceTransformer is None:
		raise ImportError(
			"sentence-transformers is required for this embedding model. "
			"Install with: pip install sentence-transformers"
		)
	if model not in _ST_MODEL_CACHE:
		_ST_MODEL_CACHE[model] = SentenceTransformer(model)
	return _ST_MODEL_CACHE[model]


# ---------------------------
# Input parsing
# ---------------------------

def _stable_chunk_id(content: str, section_ref: str, source_url: str) -> str:
	base = f"{section_ref}|{source_url}|{content}".encode("utf-8")
	return hashlib.sha256(base).hexdigest()[:32]


def _to_chunk_record(item: Dict[str, Any]) -> ChunkRecord:
	content = (item.get("content") or item.get("text") or "").strip()
	section_ref = (item.get("section_ref") or item.get("path_label") or item.get("section") or "").strip()
	source_type = str(item.get("source_type") or item.get("metadata", {}).get("source_type") or "unknown")
	authority_level = str(item.get("authority_level") or item.get("metadata", {}).get("authority_level") or "unknown")
	source_url = str(item.get("source_url") or item.get("metadata", {}).get("source_url") or "")
	provided_id = item.get("id") or item.get("chunk_id")

	if not content:
		raise ValueError("Chunk content is required")

	chunk_id = str(provided_id) if provided_id else _stable_chunk_id(content, section_ref, source_url)
	return ChunkRecord(
		content=content,
		section_ref=section_ref,
		source_type=source_type,
		authority_level=authority_level,
		source_url=source_url,
		chunk_id=chunk_id,
	)


def load_chunk_records(input_path: str) -> List[ChunkRecord]:
	with open(input_path, "r", encoding="utf-8") as f:
		raw = json.load(f)

	if isinstance(raw, list):
		items = raw
	elif isinstance(raw, dict):
		items = raw.get("base_chunks") or raw.get("chunks") or raw.get("level_chunks") or []
	else:
		raise ValueError("Unsupported input JSON structure")

	if not isinstance(items, list) or not items:
		raise ValueError("No chunks found in input file")

	records: List[ChunkRecord] = []
	skipped = 0
	for i, item in enumerate(items):
		try:
			records.append(_to_chunk_record(item))
		except Exception as exc:
			skipped += 1
			logging.warning("Skipping invalid chunk %s: %s", i, exc)

	if not records:
		raise ValueError("No valid chunks after validation")

	logging.info("Loaded %s chunks (skipped=%s)", len(records), skipped)
	return records


# ---------------------------
# Embeddings
# ---------------------------

def embed_texts(
	openai_client: Optional[OpenAI],
	texts: List[str],
	model: str = EMBEDDING_MODEL,
	max_retries: int = 4,
	retry_base_seconds: float = 1.5,
) -> List[List[float]]:
	if is_sentence_transformer_model(model):
		st_model = get_sentence_transformer(model)
		vectors = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
		return [v.tolist() for v in vectors]

	if openai_client is None:
		raise ValueError("openai_client is required for OpenAI embedding models")

	for attempt in range(max_retries + 1):
		try:
			resp = openai_client.embeddings.create(model=model, input=texts)
			vectors = [d.embedding for d in resp.data]
			if len(vectors) != len(texts):
				raise RuntimeError("Embedding response size mismatch")
			return vectors
		except Exception:
			if attempt >= max_retries:
				raise
			sleep_s = retry_base_seconds * (2 ** attempt)
			logging.warning("Embedding retry %s/%s in %.1fs", attempt + 1, max_retries, sleep_s)
			time.sleep(sleep_s)


def embed_query(openai_client: Optional[OpenAI], query: str, model: str = EMBEDDING_MODEL) -> List[float]:
	return embed_texts(openai_client, [query], model=model)[0]


# ---------------------------
# Pinecone upsert
# ---------------------------

def _iter_batches(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
	for i in range(0, len(items), batch_size):
		yield items[i : i + batch_size]


def prepare_vectors(records: List[ChunkRecord], vectors: List[List[float]]) -> List[Dict[str, Any]]:
	if len(records) != len(vectors):
		raise ValueError("records and vectors length mismatch")

	out: List[Dict[str, Any]] = []
	for rec, vec in zip(records, vectors):
		if not vec:
			raise ValueError(f"Empty vector for id={rec.chunk_id}")

		out.append(
			{
				"id": rec.chunk_id,
				"values": vec,
				"metadata": {
					"content": rec.content,
					"section_ref": rec.section_ref,
					"source_type": rec.source_type,
					"authority_level": rec.authority_level,
					"source_url": rec.source_url,
				},
			}
		)
	return out


def upsert_vectors(
	index,
	vectors: List[Dict[str, Any]],
	batch_size: int = UPSERT_BATCH_SIZE,
	namespace: str = "default",
) -> None:
	total = len(vectors)
	done = 0
	for batch in _iter_batches(vectors, batch_size):
		index.upsert(vectors=batch, namespace=namespace)
		done += len(batch)
		logging.info("Upserted %s/%s vectors", done, total)


def ingest_chunks(input_path: str, namespace: str = "default") -> None:
	openai_client = None if is_sentence_transformer_model(EMBEDDING_MODEL) else build_openai_client()
	pc = build_pinecone_client()

	records = load_chunk_records(input_path)
	probe_vec = embed_texts(openai_client, [records[0].content], model=EMBEDDING_MODEL)[0]
	vector_dim = len(probe_vec)
	ensure_index(pc, INDEX_NAME, dimension=vector_dim)
	index = get_index(pc, INDEX_NAME)

	all_vectors: List[Dict[str, Any]] = []
	for batch in _iter_batches(records, UPSERT_BATCH_SIZE):
		texts = [r.content for r in batch]
		embeddings = embed_texts(openai_client, texts, model=EMBEDDING_MODEL)
		all_vectors.extend(prepare_vectors(batch, embeddings))

	upsert_vectors(
		index=index,
		vectors=all_vectors,
		batch_size=UPSERT_BATCH_SIZE,
		namespace=namespace,
	)
	logging.info("Ingestion complete. Namespace=%s, vectors=%s", namespace, len(all_vectors))


# ---------------------------
# Retrieval + RAG
# ---------------------------

def semantic_query(
	query: str,
	top_k: int = DEFAULT_TOP_K,
	namespace: str = "default",
) -> List[Dict[str, Any]]:
	openai_client = None if is_sentence_transformer_model(EMBEDDING_MODEL) else build_openai_client()
	pc = build_pinecone_client()

	qvec = embed_query(openai_client, query, model=EMBEDDING_MODEL)
	ensure_index(pc, INDEX_NAME, dimension=len(qvec))
	index = get_index(pc, INDEX_NAME)
	candidate_k = max(top_k * 3, 20)
	result = index.query(
		vector=qvec,
		top_k=candidate_k,
		include_values=False,
		include_metadata=True,
		namespace=namespace,
	)

	result_matches = result.get("matches", []) if isinstance(result, dict) else getattr(result, "matches", [])
	matches = []
	for m in result_matches:
		m_id = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
		m_score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
		m_md = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
		matches.append(
			{
				"id": m_id,
				"score": m_score,
				"metadata": m_md or {},
			}
		)
	return rerank_matches(query=query, matches=matches, top_k=top_k)


def build_context(matches: List[Dict[str, Any]]) -> str:
	blocks: List[str] = []
	for i, item in enumerate(matches, start=1):
		md = item.get("metadata", {})
		section = md.get("section_ref", "")
		content = md.get("content", "")
		source_type = md.get("source_type", "")
		authority_level = md.get("authority_level", "")
		source_url = md.get("source_url", "")

		blocks.append(
			"\n".join(
				[
					f"[SOURCE {i}]",
					f"Section: {section}",
					f"Source Type: {source_type}",
					f"Authority Level: {authority_level}",
					f"Source URL: {source_url}",
					f"Content: {content}",
				]
			)
		)
	return "\n\n".join(blocks)


def _tokenize_for_rank(text: str) -> List[str]:
	return re.findall(r"[a-z0-9\.]+", (text or "").lower())


def _extract_cfr_ref(query: str) -> str:
	q = (query or "").lower()
	match = re.search(r"\b\d+\s*cfr\s*\d+(?:\.\d+)+\b", q)
	if match:
		return match.group(0)

	match2 = re.search(r"\b\d+(?:\.\d+)+\b", q)
	return match2.group(0) if match2 else ""


def rerank_matches(query: str, matches: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
	q_tokens = set(_tokenize_for_rank(query))
	cfr_ref = _extract_cfr_ref(query)
	authority_boost = {
		"high": 0.05,
		"medium": 0.03,
		"low": 0.01,
		"unknown": 0.0,
	}

	ranked = []
	for item in matches:
		md = item.get("metadata", {}) or {}
		content = str(md.get("content", ""))
		section_ref = str(md.get("section_ref", ""))
		source_url = str(md.get("source_url", ""))
		authority = str(md.get("authority_level", "unknown")).strip().lower()
		base = float(item.get("score") or 0.0)

		doc_tokens = set(_tokenize_for_rank(f"{section_ref} {content}"))
		overlap = len(q_tokens & doc_tokens)
		keyword_score = overlap / max(len(q_tokens), 1)

		early_text = content[:500]
		early_tokens = set(_tokenize_for_rank(f"{section_ref} {early_text}"))
		early_overlap = len(q_tokens & early_tokens)
		early_boost = 0.05 * (early_overlap / max(len(q_tokens), 1))

		citation_boost = 0.0
		if cfr_ref:
			if cfr_ref in content.lower() or cfr_ref in section_ref.lower():
				citation_boost += 0.07
			elif cfr_ref.replace(" ", "") in (content.lower() + section_ref.lower()).replace(" ", ""):
				citation_boost += 0.05

		meta_boost = 0.0
		if source_url:
			meta_boost += 0.02
		meta_boost += authority_boost.get(authority, 0.0)

		long_penalty = 0.0
		if len(content) > 2600:
			long_penalty = min((len(content) - 2600) / 10000.0, 0.08)

		final_score = (0.70 * base) + (0.25 * keyword_score) + citation_boost + meta_boost + early_boost - long_penalty
		ranked.append(
			{
				**item,
				"_debug_rank": {
					"base": round(base, 6),
					"keyword": round(keyword_score, 6),
					"early_boost": round(early_boost, 6),
					"citation_boost": round(citation_boost, 6),
					"meta_boost": round(meta_boost, 6),
					"long_penalty": round(long_penalty, 6),
					"final": round(final_score, 6),
				},
			}
		)

	ranked.sort(key=lambda x: x.get("_debug_rank", {}).get("final", 0.0), reverse=True)
	return ranked[:top_k]


def rag_answer(
	user_query: str,
	top_k: int = DEFAULT_TOP_K,
	namespace: str = "default",
	force_llm: bool = False,
) -> Dict[str, Any]:
	matches = semantic_query(query=user_query, top_k=top_k, namespace=namespace)
	context = build_context(matches)
	state = load_llm_state()
	now_ts = time.time()
	cache_key = normalize_query_for_cache(user_query)
	cached = state.get("query_cache", {}).get(cache_key)

	if cached:
		cached_ts = float(cached.get("ts", 0))
		if now_ts - cached_ts <= LLM_CACHE_TTL_SECONDS:
			return {
				"answer": cached.get("answer", ""),
				"sources": matches,
				"context": context,
				"from_cache": True,
			}

	last_llm_call_ts = float(state.get("last_llm_call_ts", 0.0))
	elapsed = now_ts - last_llm_call_ts
	if not force_llm and elapsed < LLM_MIN_INTERVAL_SECONDS:
		fallback_answer = build_retrieval_only_answer(matches)
		return {
			"answer": fallback_answer,
			"sources": matches,
			"context": context,
			"from_cache": False,
			"llm_skipped": True,
			"retry_after_seconds": int(LLM_MIN_INTERVAL_SECONDS - elapsed),
		}

	chat_client, chat_model = build_chat_client_and_model()
	system_prompt = (
		"Answer ONLY using provided sources. "
		"Cite sources like [SOURCE 1]. Do not hallucinate."
	)

	user_prompt = (
		f"Question:\n{user_query}\n\n"
		f"Retrieved Sources:\n{context}\n\n"
		"Provide a concise, accurate answer with citations."
	)

	if LLM_REQUEST_DELAY_SECONDS > 0:
		logging.info("Applying LLM delay: %.1fs", LLM_REQUEST_DELAY_SECONDS)
		time.sleep(LLM_REQUEST_DELAY_SECONDS)

	try:
		completion = chat_client.chat.completions.create(
			model=chat_model,
			temperature=0,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
		)
		answer = completion.choices[0].message.content or ""

		state["last_llm_call_ts"] = time.time()
		state.setdefault("query_cache", {})[cache_key] = {
			"answer": answer,
			"ts": state["last_llm_call_ts"],
		}
		save_llm_state(state)
	except Exception as exc:
		logging.warning("LLM call failed, returning retrieval-only answer: %s", exc)
		state["last_llm_call_ts"] = time.time()
		save_llm_state(state)
		answer = build_retrieval_only_answer(matches)
	return {
		"answer": answer,
		"sources": matches,
		"context": context,
	}


# ---------------------------
# CLI
# ---------------------------

def build_cli() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Production RAG with OpenAI + Pinecone")
	parser.add_argument("--verbose", action="store_true", help="Enable debug logs")

	sub = parser.add_subparsers(dest="command", required=True)

	ingest_p = sub.add_parser("ingest", help="Embed and upsert chunks into Pinecone")
	ingest_p.add_argument("--input", required=True, help="Path to chunks JSON")
	ingest_p.add_argument("--namespace", default="default", help="Pinecone namespace")

	query_p = sub.add_parser("query", help="Run semantic search only")
	query_p.add_argument("--question", required=True, help="User query")
	query_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
	query_p.add_argument("--namespace", default="default")

	ask_p = sub.add_parser("ask", help="Run full RAG answer")
	ask_p.add_argument("--question", required=True, help="User query")
	ask_p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
	ask_p.add_argument("--namespace", default="default")
	ask_p.add_argument(
		"--force-llm",
		action="store_true",
		help="Force LLM call even if cooldown is active",
	)

	return parser


def main() -> None:
	load_dotenv()
	parser = build_cli()
	args = parser.parse_args()
	setup_logging(verbose=args.verbose)

	if args.command == "ingest":
		ingest_chunks(input_path=args.input, namespace=args.namespace)
		return

	if args.command == "query":
		matches = semantic_query(query=args.question, top_k=args.top_k, namespace=args.namespace)
		print(json.dumps(matches, indent=2, ensure_ascii=False))
		return

	if args.command == "ask":
		result = rag_answer(
			user_query=args.question,
			top_k=args.top_k,
			namespace=args.namespace,
			force_llm=args.force_llm,
		)
		print(result["answer"])
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()
