// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif
#include <iostream>
#include <dune/common/parallel/mpihelper.hh> // An initializer of MPI
#include <dune/common/exceptions.hh> // We use exceptions

#include <stdlib.h>                                                       // include for abs function
#include <cmath>                                                          // include for sqrt, sine and cosine
#include <numeric>                                                        // include for std::inner_product and std::accumulate
#include <algorithm>                                                      // include for std::transform
#include <iterator>                                                       // include for std::back_inserter
#include <vector>                                                         // include for std::vector
#include <dune/common/parametertreeparser.hh>                             // include for ParameterTree
#include <dune/common/fvector.hh>                                         // include for FieldVector data structure
#include <dune/duneuro-analytic-solution/duneuro-analytic-solution.hh>    // include for analytic MEG solution in sphere models
#include <duneuro/io/point_vtk_writer.hh>                                 // include for creating vtk-files
#include <duneuro/io/dipole_reader.hh>                                    // include for reading dipoles from file
#include <duneuro/driver/driver_factory.hh>                               // include for the driver interface

// compute euclidean norm of a vector
template<class T>
T norm(const std::vector<T>& vector) {
  return std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), T(0.0)));
}

// compute relative error
template<class T>
  T relative_error(const std::vector<T>& numerical_solution, const std::vector<T>& analytical_solution) {
  std::vector<T> diff;
  std::transform(numerical_solution.begin(), 
                 numerical_solution.end(), 
                 analytical_solution.begin(), 
                 std::back_inserter(diff),
                 [] (const T& num_val, const T& ana_val) {return num_val - ana_val;});
   return norm(diff) / norm(analytical_solution);
}

// compute MAG error
template<class T>
T magnitude_error(const std::vector<T>& numerical_solution, const std::vector<T>& analytical_solution) {
  return norm(numerical_solution) / norm(analytical_solution);
}

// compute RDM error
template<class T>
T relative_difference_measure(const std::vector<T>& numerical_solution, const std::vector<T>& analytical_solution) {
  T norm_numerical = norm(numerical_solution);
  T norm_analytical = norm(analytical_solution);
  std::vector<T> diff;
  std::transform(numerical_solution.begin(),
                 numerical_solution.end(),
                 analytical_solution.begin(),
                 std::back_inserter(diff),
                 [norm_numerical, norm_analytical] (const T& num_val, const T& ana_val)
                   {return num_val / norm_numerical - ana_val / norm_analytical;});
  return norm(diff); 
}

int main(int argc, char** argv)
{
  try{
    // Maybe initialize MPI
    Dune::MPIHelper& helper = Dune::MPIHelper::instance(argc, argv);

    constexpr size_t dim = 3;
    using FieldType = double;
    using Coordinate = Dune::FieldVector<FieldType, dim>;
    
    // read parameter tree
    std::cout << " Reading parameter tree\n";
    Dune::ParameterTree config_tree;
    Dune::ParameterTreeParser config_parser;
     
    config_parser.readINITree("configs.ini", config_tree);
    std::cout << " Parameter tree read\n";

    /////////////////////////////////////////
    // create coils
    /////////////////////////////////////////

    std::cout << " Creating coils\n";

    // we first create some artificial coil positions around the sphere
    Coordinate center;
    center[0] = 127;
    center[1] = 127;
    center[2] = 127;
    
    size_t polar_frequency = config_tree.get<size_t>("input.polar_frequency");
    size_t azimuth_frequency = config_tree.get<size_t>("input.azimuth_frequency");
    size_t number_of_coils = (polar_frequency - 1) * azimuth_frequency;
    
    FieldType radius = config_tree.get<FieldType>("input.radius");
    FieldType pi = 3.14159265358979323846;
    
    Coordinate current;
    std::vector<Coordinate> coils(number_of_coils);
    for(size_t i = 0; i < azimuth_frequency; ++i) {
      for(size_t j = 1; j < polar_frequency; ++j) {
        current[0] = radius * std::sin(j * (pi / polar_frequency)) * std::cos(i * (2 * pi / azimuth_frequency));
        current[1] = radius * std::sin(j * (pi / polar_frequency)) * std::sin(i * (2 * pi / azimuth_frequency));
        current[2] = radius * std::cos(j * (pi / polar_frequency));
        
        current += center;
        
        coils[i * (polar_frequency - 1) + j - 1] = current;
      }
    }
    
    // we now create accompanying projections, where we always choose the three unit vectors
    std::vector<std::vector<Coordinate>> projections(number_of_coils);
    for(size_t i = 0; i < number_of_coils; ++i) {
      Coordinate dummy;
      for(size_t j = 0; j < dim; ++j) {
        dummy = 0.0;
        dummy[j] = 1; 
        projections[i].push_back(dummy);
      }
    }
    
    size_t number_of_fluxes = number_of_coils * dim;
    
    std::cout << " Coils created\n";
    
    /////////////////////////////////////////
    // read dipoles
    /////////////////////////////////////////
    
    // read dipoles from file
    std::cout << " Reading dipoles\n";
    std::string filename_dipoles = config_tree.get<std::string>("input.dipoles_filename");
    std::vector<duneuro::Dipole<FieldType, dim>> dipoles = duneuro::DipoleReader<FieldType, dim>::read(filename_dipoles);
    std::cout << " Dipoles read\n";
    duneuro::Dipole test_dipole = dipoles[0];
    
    /////////////////////////////////////////
    // solve MEG forward problem numerically
    /////////////////////////////////////////

    std::cout << " Solving MEG forward problem numerically\n";

    // create driver
    std::cout << " Creating driver\n";
    using Driver = duneuro::DriverInterface<dim>;
    std::unique_ptr<Driver> driver_ptr = duneuro::DriverFactory<dim>::make_driver(config_tree);
    std::cout << " Driver created\n";
    
    // solve EEG forward problem
    std::cout << " Solving EEG forward problem\n";
    auto eeg_solution_storage_ptr = driver_ptr->makeDomainFunction();
    driver_ptr->solveEEGForward(test_dipole, *eeg_solution_storage_ptr, config_tree);
    std::cout << " EEG forward problem solved\n";
    
    // solve MEG forward problem
    std::cout << " Solving MEG forward problem\n";
    driver_ptr->setCoilsAndProjections(coils, projections);
    std::vector<FieldType> numeric_solution = driver_ptr->solveMEGForward(*eeg_solution_storage_ptr, config_tree);
    std::cout << " MEG forward problem solved\n";
    
    std::cout << " Numerical solution computed\n";
    
    /////////////////////////////////////////
    // solve MEG forward problem analytically
    /////////////////////////////////////////
    std::cout << " Solving problem analytically\n";
    
    duneuro::AnalyticSolutionMEG analyticSolution(center);
    analyticSolution.bind(test_dipole);
    
    std::vector<FieldType> analytic_solution(number_of_fluxes);
    for(size_t i = 0; i < number_of_coils; ++i) {
      for(size_t j = 0; j < dim; ++j) {
        analytic_solution[i*dim + j] = analyticSolution.secondaryField(coils[i], projections[i][j]);
      }
    }

    std::cout << " Analytical solution computed\n";

    /////////////////////////////////////////
    // comparision of analytic and numeric solution
    /////////////////////////////////////////
    
    std::cout << "\n We now compare the analytical and the numerical solution\n";
    std::cout << " Norm analytic solution :\t\t " << norm(analytic_solution) << "\n";
    std::cout << " Norm numeric solution :\t\t " << norm(numeric_solution) << "\n";
    std::cout << " Relative error :\t\t\t " << relative_error(numeric_solution, analytic_solution) << "\n";
    std::cout << " MAG : \t\t\t\t\t " << magnitude_error(numeric_solution, analytic_solution) << "\n";
    std::cout << " RDM : \t\t\t\t\t " << relative_difference_measure(numeric_solution, analytic_solution) << "\n";
    
    std::cout << " Comparison finished\n\n";

    /////////////////////////////////////////
    // visualization
    /////////////////////////////////////////
    
    if(config_tree.get<bool>("output.visualize_fields")) {
      std::cout << " Writing output\n";
      // first write volume conductor
      std::cout << " Writing volume conductor\n";
      auto volume_writer_ptr = driver_ptr->volumeConductorVTKWriter(config_tree);
      volume_writer_ptr->addVertexData(*eeg_solution_storage_ptr, "correction_potential");
      volume_writer_ptr->addCellDataGradient(*eeg_solution_storage_ptr, "gradient");
      config_tree["output.filename"] = config_tree["output.filename_volume_conductor"];
      volume_writer_ptr->write(config_tree.sub("output"));

      // compute fields analytically
      std::vector<Coordinate> magnetic_field(number_of_coils);
      std::vector<Coordinate> primary_field(number_of_coils);
      std::vector<Coordinate> secondary_field(number_of_coils);
      for(size_t i = 0; i < number_of_coils; ++i) {
        magnetic_field[i] = analyticSolution.totalField(coils[i]);
        primary_field[i] = analyticSolution.primaryField(coils[i]);
        secondary_field[i] = analyticSolution.secondaryField(coils[i]);
      } 

      // write dipole
      std::cout << " Writing dipole\n";
      duneuro::PointVTKWriter<FieldType, dim> dipole_writer{test_dipole};
      std::string dipole_filename_string = config_tree.get<std::string>("output.filename_dipole");
      dipole_writer.write(dipole_filename_string);

      // write magnetic fields
      std::cout << " Writing magnetic field\n";
      duneuro::PointVTKWriter<FieldType, dim> field_writer{coils};
      field_writer.addVectorData("total_magnetic_field_analytic", magnetic_field);
      field_writer.addVectorData("primary_magnetic_field_analytic", primary_field);
      field_writer.addVectorData("secondary_magnetic_field_analytic", secondary_field);
      std::string magnetic_field_filename_string = config_tree.get<std::string>("output.filename_magnetic_fields_analytic");
      field_writer.write(magnetic_field_filename_string);
      
      std::cout << "Output written\n";
    }

    std::cout << " The program didn't crash!\n";

    return 0;
  }
  catch (Dune::Exception &e){
    std::cerr << "Dune reported error: " << e << std::endl;
  }
  catch (...){
    std::cerr << "Unknown exception thrown!" << std::endl;
  }
}
