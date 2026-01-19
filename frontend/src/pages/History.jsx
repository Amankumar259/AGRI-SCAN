import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Leaf, Clock, TrendingUp, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const History = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API}/predictions/history?limit=20`);
      if (!response.ok) {
        throw new Error("Failed to fetch history");
      }
      const data = await response.json();
      setHistory(data.results || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence > 0.8) return "bg-green-100 text-green-800 border-green-300";
    if (confidence > 0.5)
      return "bg-yellow-100 text-yellow-800 border-yellow-300";
    return "bg-red-100 text-red-800 border-red-300";
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="min-h-screen bg-[#F4F5F0] py-8 px-4"
    >
      <div className="container mx-auto max-w-6xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              data-testid="back-home-btn"
              variant="outline"
              onClick={() => navigate("/")}
              className="border-[#1A3C34] text-[#1A3C34] hover:bg-[#1A3C34] hover:text-white"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
            <div className="flex items-center gap-2">
              <Clock className="w-8 h-8 text-[#1A3C34]" />
              <h1
                className="text-3xl font-bold text-[#1A3C34]"
                style={{ fontFamily: "'Playfair Display', serif" }}
              >
                Detection History
              </h1>
            </div>
          </div>
          <Button
            data-testid="new-scan-btn"
            onClick={() => navigate("/dashboard")}
            className="bg-[#1A3C34] text-white hover:bg-[#1A3C34]/90 rounded-none border-b-2 border-[#D4FF32] px-6 py-3 uppercase tracking-wider"
            style={{ fontFamily: "'JetBrains Mono', monospace" }}
          >
            New Scan
          </Button>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-white border border-[#1A3C34]/10 rounded-sm result-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p
                    className="text-sm text-[#5C6B66] uppercase tracking-wide"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    Total Scans
                  </p>
                  <p
                    className="text-3xl font-bold text-[#1A3C34] mt-1"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {history.length}
                  </p>
                </div>
                <Leaf className="w-12 h-12 text-[#D4FF32]" />
              </div>
            </CardContent>
          </Card>
          <Card className="bg-white border border-[#1A3C34]/10 rounded-sm result-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p
                    className="text-sm text-[#5C6B66] uppercase tracking-wide"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    Avg Confidence
                  </p>
                  <p
                    className="text-3xl font-bold text-[#1A3C34] mt-1"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {history.length > 0
                      ? (
                          (history.reduce((sum, h) => sum + h.confidence, 0) /
                            history.length) *
                          100
                        ).toFixed(0)
                      : 0}
                    %
                  </p>
                </div>
                <TrendingUp className="w-12 h-12 text-[#E07A5F]" />
              </div>
            </CardContent>
          </Card>
          <Card className="bg-white border border-[#1A3C34]/10 rounded-sm result-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p
                    className="text-sm text-[#5C6B66] uppercase tracking-wide"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    Issues Found
                  </p>
                  <p
                    className="text-3xl font-bold text-[#1A3C34] mt-1"
                    style={{ fontFamily: "'JetBrains Mono', monospace" }}
                  >
                    {
                      history.filter((h) => h.predicted_disease !== "Healthy")
                        .length
                    }
                  </p>
                </div>
                <AlertCircle className="w-12 h-12 text-[#E07A5F]" />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* History List */}
        <Card className="bg-white border border-[#1A3C34]/10 rounded-sm">
          <CardHeader>
            <CardTitle
              className="text-[#1A3C34]"
              style={{ fontFamily: "'Playfair Display', serif" }}
            >
              Recent Predictions
            </CardTitle>
            <CardDescription className="text-[#5C6B66]">
              View all your previous plant disease scans
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-12 text-[#5C6B66]">
                <Leaf className="w-12 h-12 mx-auto mb-4 animate-pulse" />
                <p>Loading history...</p>
              </div>
            ) : error ? (
              <div className="text-center py-12 text-red-600">
                <AlertCircle className="w-12 h-12 mx-auto mb-4" />
                <p>{error}</p>
              </div>
            ) : history.length === 0 ? (
              <div className="text-center py-12 text-[#8D9995]">
                <Leaf className="w-16 h-16 mx-auto mb-4 opacity-30" />
                <p className="text-lg mb-2">No predictions yet</p>
                <p className="text-sm">
                  Start by scanning your first plant image
                </p>
                <Button
                  onClick={() => navigate("/dashboard")}
                  className="mt-4 bg-[#1A3C34] text-white hover:bg-[#1A3C34]/90 rounded-none border-b-2 border-[#D4FF32] px-6 py-3"
                >
                  Start Scanning
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                {history.map((item, index) => (
                  <motion.div
                    key={item.prediction_id || index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="specimen-tag bg-[#F4F5F0] rounded-sm p-4 hover:bg-[#1A3C34]/5 transition-colors"
                    data-testid={`history-item-${index}`}
                  >
                    <div className="flex items-center justify-between flex-wrap gap-4">
                      <div className="flex-1 min-w-[200px]">
                        <p className="font-semibold text-[#1A3C34] mb-1">
                          {item.filename}
                        </p>
                        <p
                          className="text-xs text-[#5C6B66]"
                          style={{ fontFamily: "'JetBrains Mono', monospace" }}
                        >
                          {formatDate(item.timestamp)}
                        </p>
                      </div>
                      <div className="flex items-center gap-4">
                        <div>
                          <p
                            className="text-sm text-[#5C6B66] uppercase tracking-wide mb-1"
                            style={{
                              fontFamily: "'JetBrains Mono', monospace",
                            }}
                          >
                            Disease
                          </p>
                          <Badge
                            className={`${
                              item.predicted_disease === "Healthy"
                                ? "bg-green-100 text-green-800 border-green-300"
                                : "bg-red-100 text-red-800 border-red-300"
                            } border`}
                          >
                            {item.predicted_disease}
                          </Badge>
                        </div>
                        <div>
                          <p
                            className="text-sm text-[#5C6B66] uppercase tracking-wide mb-1"
                            style={{
                              fontFamily: "'JetBrains Mono', monospace",
                            }}
                          >
                            Confidence
                          </p>
                          <Badge
                            className={`${getConfidenceColor(
                              item.confidence
                            )} border font-mono`}
                          >
                            {(item.confidence * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  );
};

export default History;
