import React from "react";
import { motion } from "framer-motion";
import { Check, AlertTriangle, Activity } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

const PredictionResult = ({ prediction }) => {
  if (!prediction) return null;

  const isHealthy = prediction.predicted_disease === "Healthy";
  const confidencePercent = (prediction.confidence * 100).toFixed(1);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card
        data-testid="prediction-result"
        className="bg-white border border-[#1A3C34]/10 rounded-sm result-card"
      >
        <CardHeader>
          <CardTitle
            className="flex items-center gap-2 text-[#1A3C34]"
            style={{ fontFamily: "'Playfair Display', serif" }}
          >
            {isHealthy ? (
              <Check className="h-6 w-6 text-green-600" />
            ) : (
              <AlertTriangle className="h-6 w-6 text-red-600" />
            )}
            Prediction Result
          </CardTitle>
          <CardDescription className="text-[#5C6B66]">
            Disease classification analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Primary Diagnosis */}
          <div className="specimen-tag bg-[#F4F5F0] p-6 rounded-sm">
            <p
              className="text-xs text-[#5C6B66] uppercase tracking-widest mb-2"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              Primary Diagnosis
            </p>
            <p
              data-testid="predicted-disease"
              className="text-3xl font-bold text-[#1A3C34] mt-1"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              {prediction.predicted_disease}
            </p>
          </div>

          {/* Confidence Score */}
          <div className="bg-gradient-to-r from-[#1A3C34]/5 to-[#D4FF32]/10 p-6 rounded-sm">
            <div className="flex items-center justify-between mb-3">
              <p
                className="text-sm font-medium text-[#1A3C34] uppercase tracking-wide"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                Confidence Score
              </p>
              <span
                data-testid="confidence-score"
                className="text-2xl font-bold text-[#1A3C34]"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                {confidencePercent}%
              </span>
            </div>
            <div className="relative w-full h-3 bg-[#1A3C34]/10 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidencePercent}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="confidence-bar h-full"
              />
            </div>
          </div>

          {/* All Predictions */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-[#1A3C34]" />
              <p
                className="text-sm font-medium text-[#1A3C34] uppercase tracking-wide"
                style={{ fontFamily: "'JetBrains Mono', monospace" }}
              >
                Top Predictions
              </p>
            </div>
            {prediction.all_predictions?.map((pred, idx) => (
              <div
                key={idx}
                data-testid={`prediction-item-${idx}`}
                className="flex justify-between items-center p-3 bg-[#F4F5F0] rounded-sm hover:bg-[#1A3C34]/5 transition-colors"
              >
                <span className="text-sm text-[#1A3C34] font-medium">
                  {pred.class}
                </span>
                <Badge
                  variant="secondary"
                  className="bg-[#1A3C34] text-white font-mono text-xs"
                >
                  {(pred.confidence * 100).toFixed(1)}%
                </Badge>
              </div>
            ))}
          </div>

          {/* Warning Message */}
          {!isHealthy && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-red-50 border-l-4 border-red-500 p-4 rounded-sm"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-red-900 mb-1">
                    Disease Detected
                  </p>
                  <p className="text-sm text-red-800">
                    This plant shows signs of disease. Please consult with
                    agricultural experts for proper treatment and management
                    strategies.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Healthy Message */}
          {isHealthy && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="bg-green-50 border-l-4 border-green-500 p-4 rounded-sm"
            >
              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-semibold text-green-900 mb-1">
                    Plant Looks Healthy!
                  </p>
                  <p className="text-sm text-green-800">
                    No diseases detected. Continue with regular care and
                    monitoring to maintain plant health.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Metadata */}
          <div className="pt-4 border-t border-[#1A3C34]/10">
            <p
              className="text-xs text-[#8D9995]"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              Analysis ID: {prediction.prediction_id}
            </p>
            <p
              className="text-xs text-[#8D9995] mt-1"
              style={{ fontFamily: "'JetBrains Mono', monospace" }}
            >
              Timestamp: {new Date(prediction.timestamp).toLocaleString()}
            </p>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default PredictionResult;
