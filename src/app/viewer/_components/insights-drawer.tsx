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

function InsightsDrawer() {
    const fractureInsights: string[] = [
        "The fracture's location and severity determine the best treatment approach.",
        "3D visualization helps assess misalignment and healing progress.",
        "Bone density affects fracture risk and recovery time.",
        "Some fractures require surgery, while others heal with immobilization.",
        "Repetitive stress can lead to microfractures before a full break occurs.",
    ];

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

                    {fractureInsights.map((fi, idx) => (
                        <p key={idx}>{fi}</p>
                    ))}

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
