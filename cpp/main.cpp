// cpp.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "simulator.h"
#include <iostream>

int main()
{
    //IsingModel isingModel(2);
    Node node = "a";
    LinearBiases linear = { {node, 0.e0} };
    Edge edge = std::make_pair("a", "b");
    QuadraticBiases quadratic = { {edge, 1.e0} };
    IsingModel isingModel(linear, quadratic);
    
    isingModel.SetTemperature(10.e0);
    isingModel.ChangeAlgorithmTo(IsingModel::Algorithm::SCA);
    isingModel.Update();
    std::cout << isingModel.GetEnergy() << std::endl;
    std::cout << "{ ";
    for (auto spin : isingModel.GetSpins())
        std::cout << "{" << std::get<std::string>(spin.first) << ": " << spin.second << "}, ";
    std::cout << "}" << std::endl;
    isingModel.Print();
    return 0;
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
