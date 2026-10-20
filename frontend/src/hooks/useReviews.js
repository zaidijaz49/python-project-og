import { useEffect, useState } from "react";
import { api } from "../api/client";

const STORAGE_KEY = "techmart-reviews-v1";
export const MAX_REVIEW_LENGTH = 1000; // backend allows 10,000; reviews stay short

// A few starter reviews so the page doesn't look empty (already "analyzed").
const SEED_REVIEWS = {
  1: [
    {
      id: "seed-1",
      text: "Best headphones in this price. The bass is amazing!",
      summary: "The customer loves the sound and the value for money.",
      sentiment: "positive",
      tags: ["sound quality", "value"],
    },
  ],
  6: [
    {
      id: "seed-2",
      text: "Keys started sticking after two weeks.",
      summary: "Keys became sticky after a short time.",
      sentiment: "negative",
      tags: ["durability", "keys"],
    },
  ],
};

function loadReviews() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? SEED_REVIEWS;
  } catch {
    return SEED_REVIEWS;
  }
}

export function validateReview(text) {
  const trimmed = text.trim();
  if (!trimmed) return "Please write something first.";
  if (trimmed.length < 5) return "Please write a little more (at least 5 characters).";
  if (trimmed.length > MAX_REVIEW_LENGTH)
    return `Please keep it under ${MAX_REVIEW_LENGTH} characters.`;
  return null;
}

// How many positive / neutral / negative reviews a product has.
export function countSentiments(list = []) {
  const counts = { positive: 0, neutral: 0, negative: 0 };
  list.forEach((review) => (counts[review.sentiment] += 1));
  return counts;
}

export function useReviews() {
  const [reviews, setReviews] = useState(loadReviews);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(reviews));
    } catch {
      // Storage blocked: reviews still work for this visit.
    }
  }, [reviews]);

  // Sends the review to /analyze, saves it with the AI result, and returns it.
  // Throws ApiError (with a friendly message) if the analysis fails.
  async function submitReview(productId, text) {
    const clean = text.trim();
    const analysis = await api.analyze(clean); // { summary, sentiment, tags }
    const review = { id: `${Date.now()}`, text: clean, ...analysis };
    setReviews((all) => ({ ...all, [productId]: [review, ...(all[productId] ?? [])] }));
    return review;
  }

  return { reviews, submitReview };
}
