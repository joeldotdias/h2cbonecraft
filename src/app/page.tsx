import { Hero } from "./_components/hero";
import { ModeToggle } from "./_components/theme-toggle";

export default function Home() {
    return (
        <main>
            <div className="flex justify-between items-center px-4 py-4">
                <div></div>
                <ModeToggle />
            </div>
            <Hero
                heading="BoneCraft"
                description="Give me 2D, I give you 3D. Laxmi Chit Fund type shit"
                // description="Get 3D Bone structures from 2D images"
                image="/logo.jpeg"
            />
        </main>
    );
}
