import { Button } from "@/components/ui/button";
import {
    Drawer,
    DrawerClose,
    DrawerContent,
    DrawerFooter,
    DrawerHeader,
    DrawerTitle,
    DrawerTrigger,
} from "@/components/ui/drawer";
import { useEffect, useState } from "react";
import { toast } from "sonner";

function InsightsDrawer() {
    const insights: string = JSON.parse(
        localStorage.getItem("xrayInsights") || "{}",
    );
    const spinsights = insights.split("FRACTURE ASSESSMENT")[1].trim();
    const actual = spinsights.replace("-----", "\n");
    console.log(insights);

    /* const [insights, setInsights] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [alreadyFetched, setAlreadyFetched] = useState(false);

    useEffect(() => {
        fetchInsights();
    }, []);

    const fetchInsights = async () => {
        if (alreadyFetched || loading) {
            return;
        }

        const originalImage = sessionStorage.getItem("originalImage");
        const fileInfo = JSON.parse(sessionStorage.getItem("fileInfo") || "{}");

        if (!originalImage) {
            toast.error("Didn't get back a thing");
            return;
        }

        setLoading(true);

        try {
            // Convert base64 string back to a Blob
            const fetchResponse = await fetch(originalImage);
            const blob = await fetchResponse.blob();

            // Get file info if available
            const fileInfo = JSON.parse(
                sessionStorage.getItem("fileInfo") || "{}",
            );
            const file = new File([blob], fileInfo.name || "image.jpg", {
                type: fileInfo.type || blob.type,
            });

            // Create FormData and append the file
            const formData = new FormData();
            formData.append("xray", file);

            // Send to second API
            const response = await fetch("http://localhost:5000/insights", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            // Update insights with API response
            setInsights(
                data.insights || "No insights available for this X-ray.",
            );
            setAlreadyFetched(true);
        } catch (error) {
            console.error("Error analyzing image:", error);
            setInsights(
                "Could not analyze this X-ray. Please try again later.",
            );
        } finally {
            setLoading(false);

            // Clean up storage after use
            sessionStorage.removeItem("originalImage");
            sessionStorage.removeItem("fileInfo");
        }
    }; */

    /* const fractureInsights: string[] = [
        "The fracture's location and severity determine the best treatment approach.",
        "3D visualization helps assess misalignment and healing progress.",
        "Bone density affects fracture risk and recovery time.",
        "Some fractures require surgery, while others heal with immobilization.",
        "Repetitive stress can lead to microfractures before a full break occurs.",
    ]; */

    // {
    /* {fractureInsights.map((fi, idx) => (
                        <p key={idx}>{fi}</p>
                    ))} */
    // }
    return (
        <Drawer>
            <DrawerTrigger asChild className="w-24">
                <Button variant="outline" className="mt-1 ml-2">
                    Show Insights
                </Button>
            </DrawerTrigger>
            <DrawerContent>
                <div className="mx-auto w-full max-w-sm">
                    <DrawerHeader>
                        <DrawerTitle>Xray Insights</DrawerTitle>
                    </DrawerHeader>

                    <p>{actual || "No insights"}</p>

                    <DrawerFooter>
                        <Button>Save</Button>
                        <DrawerClose asChild>
                            <Button variant="outline">Close</Button>
                        </DrawerClose>
                    </DrawerFooter>
                </div>
            </DrawerContent>
        </Drawer>
    );
}

export { InsightsDrawer };
