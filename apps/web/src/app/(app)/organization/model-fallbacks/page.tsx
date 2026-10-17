import type { Metadata } from "next";

import { ModelFallbacksPage } from "./_page-component";

export const metadata: Metadata = {
    title: "Model fallbacks",
    openGraph: { title: "Model fallbacks" },
};

export default function ModelFallbacksSettingsPage() {
    return <ModelFallbacksPage />;
}
