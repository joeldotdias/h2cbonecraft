"use client";

import { useSearchParams } from "next/navigation";
import * as THREE from "three";
import React, { useEffect, useRef } from "react";
import { Canvas, useLoader } from "@react-three/fiber";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { toast } from "sonner";
import { InsightsDrawer } from "./_components/insights-drawer";

function BoneCompi({ objUrl }: { objUrl: string }) {
    const meshRef = useRef<THREE.Mesh>(null!);
    const obj = useLoader(OBJLoader, objUrl);

    useEffect(() => {
        if (meshRef.current) {
            meshRef.current.position.set(0, 0, 0);

            const box = new THREE.Box3().setFromObject(meshRef.current);
            const center = box.getCenter(new THREE.Vector3());

            meshRef.current.position.x = -center.x;
            meshRef.current.position.y = -center.y;
            meshRef.current.position.z = -center.z;

            let material = new THREE.MeshStandardMaterial({
                color: 0xbfab52,
                roughness: 0.65,
                metalness: 0.1,
                bumpScale: 0.05,
                flatShading: false,
            });

            obj.traverse((child) => {
                if (child instanceof THREE.Mesh) {
                    child.userData.originalMaterial = child.material;
                    child.material = material;
                }
            });
        }
    }, [obj]);

    return (
        <mesh ref={meshRef}>
            <primitive object={obj} />
        </mesh>
    );
}

function Boneto({ objUrl }: { objUrl: string }) {
    return (
        <div className="flex justify-center items-center h-screen">
            <Canvas>
                <PerspectiveCamera makeDefault position={[0, 0, 5]} />
                <OrbitControls
                    enablePan={true}
                    enableZoom={true}
                    enableRotate={true}
                />
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <hemisphereLight intensity={0.3} />
                <BoneCompi objUrl={objUrl} />
            </Canvas>
        </div>
    );
}

function Viewer() {
    const searchParams = useSearchParams();
    const objUrl = searchParams.get("objUrl");
    if (!objUrl) {
        toast.error("Didn't get back a thing");
        return;
    }

    return (
        <main>
            <InsightsDrawer />
            <Boneto objUrl={objUrl} />
        </main>
    );
}

export default Viewer;
