"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import React, { useState } from "react";
import { toast } from "sonner";
import { useRouter } from "next/navigation";

type HeroProps = {
    heading: string;
    description: string;
    image: string;
};

function Hero({ heading, description, image }: HeroProps) {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const router = useRouter();

    const fileBeenChanged = (ev: React.ChangeEvent<HTMLInputElement>) => {
        if (!ev.target.files) {
            toast.error("Couldn't upload file");
            return;
        }

        const file = ev.target.files[0];
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));
        toast.success("Uploaded file");
        console.log("Selected: ", file);
    };

    const pleaseUpload = async () => {
        if (!selectedFile) {
            toast.error("No file selected");
            return;
        }

        const formData = new FormData();
        formData.append("xray", selectedFile);

        try {
            const response = await fetch("http://localhost:5000/upload", {
                method: "POST",
                body: formData,
            });

            const bob = await response.blob();
            const objUrl = URL.createObjectURL(bob);

            toast.success("Processed your xray");

            const insightsResponse = await fetch(
                "http://localhost:5000/analyze",
                {
                    method: "POST",
                    body: formData, // You can reuse the same FormData or send the objUrl
                },
            );

            if (!insightsResponse.ok) {
                toast.error("No insightsss wahhh");
            }
            const { insights } = await insightsResponse.json();
            console.log(insights);
            localStorage.setItem("xrayInsights", JSON.stringify(insights));

            router.push(`/viewer?objUrl=${encodeURIComponent(objUrl)}`);
        } catch (err) {
            console.error(err);
            toast.error("Couldn't process xray");
        }
    };

    return (
        <section className="py-28 px-4">
            <div className="container">
                <div className="grid items-center gap-8 lg:grid-cols-2">
                    <div className="flex flex-col items-center text-center lg:items-start lg:text-left">
                        <h1 className="my-6 text-pretty text-4xl font-bold lg:text-6xl">
                            {heading}
                        </h1>
                        <p className="mb-8 max-w-xl text-muted-foreground lg:text-xl">
                            {description}
                        </p>
                        <div className="flex w-full flex-col justify-center gap-2 sm:flex-row lg:justify-start">
                            <div className="grid max-w-sm items-center gap-1.5">
                                <Input
                                    id="xray"
                                    type="file"
                                    onChange={fileBeenChanged}
                                />
                            </div>

                            <Button
                                variant="outline"
                                className="w-full sm:w-auto"
                                onClick={pleaseUpload}
                            >
                                Upload
                            </Button>
                        </div>
                        {preview && (
                            <div className="mt-4">
                                <img
                                    src={preview}
                                    alt="Preview"
                                    className="max-w-xs rounded-md shadow-md"
                                />
                            </div>
                        )}
                    </div>
                    <img
                        src={image}
                        alt="Site Logo"
                        className="max-h-96 w-full rounded-md object-cover"
                    />
                </div>
            </div>
        </section>
    );
}

export { Hero };
